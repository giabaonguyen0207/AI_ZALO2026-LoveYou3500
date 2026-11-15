"""Offline seq2seq correction as an advisory variant; the request stays immutable."""
from collections import Counter, OrderedDict
from difflib import SequenceMatcher
import re
import time
import unicodedata

from ..agent.language import abbreviation_hints, folded
from ..models.artifacts import verify_artifact
from .contracts import RetrievalFatalError


def words(text):
    return re.findall(r'\w+', folded(text))


def validate_variant(original, candidate):
    """Reject altered literals/polarity/measurement intent, not just bad spelling."""
    if not isinstance(candidate, str) or not candidate.strip():
        return 'empty'
    candidate = unicodedata.normalize('NFC', candidate.strip())
    if len(candidate) > max(64, len(original)*1.8):
        return 'length_changed'
    # Preserve numeric spellings including dates, signs, percentages and codes.
    literals = r'(?<!\w)[+-]?\d[\w.,:/%+-]*|\b\w*[\d_]\w*\b'
    if Counter(re.findall(literals, original)) != Counter(re.findall(literals, candidate)):
        return 'numbers_or_codes_changed'
    if Counter(re.findall(r'<=|>=|!=|==|[<>=%≤≥]', original)) != Counter(re.findall(r'<=|>=|!=|==|[<>=%≤≥]', candidate)):
        return 'comparison_symbol_changed'
    protected = re.findall(r'[`"“]([^`"”]+)[`"”]', original)
    protected += re.findall(r'\b[A-ZĐ][\w-]*(?:\s+[A-ZĐ][\w-]*)*', original)
    # Sentence-initial generic request words are not entity names.
    generic = {'tong','doanh','bao','cho','tinh','so','san','khach','don','trong','thang','nam','ngay','top'}
    for term in protected:
        if len(term.split()) == 1 and folded(term) in generic:
            continue
        if not re.search(r'(?<!\w)'+re.escape(term)+r'(?!\w)', candidate, re.I):
            return 'quoted_or_named_entity_changed'
    before, after = words(original), words(candidate)
    polarity = {'ko':'khong','k':'khong','hong':'khong','hông':'khong'}
    def negatives(tokens):
        return Counter(polarity.get(t,t) for t in tokens if polarity.get(t,t) in {'khong','chua','tru','ngoai'})
    if negatives(before) != negatives(after):
        return 'negation_changed'
    concepts = (
        ('doanh thu','dthu','revenue'), ('lai gop','loi nhuan','lai rong'),
        ('so luong','sl'), ('lon hon','tren'), ('nho hon','duoi'),
        ('nhieu nhat','cao nhat','ban chay'), ('it nhat','thap nhat'),
        ('tuan','thang','nam','ngay'), ('kg','gam','lit','ml','thung','hop','chai','cai'),
    )
    # Keep individual time/unit/metric concepts, allowing only declared aliases.
    for group in concepts:
        if group in (concepts[-2], concepts[-1]):
            subgroups = [(term,) for term in group]
        else:
            subgroups = [group]
        for aliases in subgroups:
            def count(tokens):
                text=' '+' '.join(tokens)+' '
                return sum(text.count(' '+term+' ') for term in aliases)
            if count(before) != count(after):
                return 'metric_period_unit_or_comparison_changed'
    if SequenceMatcher(None,' '.join(before),' '.join(after)).ratio() < 0.55:
        return 'rewrite_too_large'
    return None


class VietnameseCorrector:
    def __init__(self, settings):
        self.settings = settings
        self.model = self.tokenizer = None
        self.cache = OrderedDict()
        self.manifest = verify_artifact(settings.correction_dir, settings.correction_key)
        self.calls = self.cache_hits = 0

    def load(self):
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            dtype = self.settings.dtype
            if dtype == 'auto':
                dtype = 'float16' if self.settings.device.startswith('cuda') else 'float32'
            tokenizer = AutoTokenizer.from_pretrained(self.settings.correction_dir,
                local_files_only=True, trust_remote_code=False)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.settings.correction_dir,
                local_files_only=True, trust_remote_code=False, dtype=getattr(torch,dtype))
            model = model.to(self.settings.device).eval()
            self.torch, self.tokenizer, self.model = torch, tokenizer, model
        except Exception as exc:
            raise RetrievalFatalError(f'Cannot initialize Vietnamese correction: {exc}') from exc

    def _generate(self, text, deadline):
        from transformers import StoppingCriteria, StoppingCriteriaList
        class Deadline(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return time.monotonic() >= deadline
        tokens = self.tokenizer(text, return_tensors='pt', truncation=False)
        if tokens['input_ids'].shape[-1] > self.settings.correction_max_tokens:
            return None, 'input_too_long'
        tokens = tokens.to(self.settings.device)
        with self.torch.inference_mode():
            output = self.model.generate(**tokens, do_sample=False, num_beams=1,
                max_new_tokens=self.settings.correction_max_tokens,
                stopping_criteria=StoppingCriteriaList([Deadline()]))
        if time.monotonic() >= deadline:
            return None, 'timeout'
        ids = output[0].tolist()
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos,int) else eos or []
        if not ids or ids[-1] not in eos or len(ids)-1 >= self.settings.correction_max_tokens:
            return None, 'incomplete_output'
        return self.tokenizer.decode(ids,skip_special_tokens=True).strip(), None

    def correct(self, text, remaining_seconds):
        started = time.monotonic()
        event = {'phase':'question_correction','model':self.settings.correction_key,
                 'original':text,'advisory_only':True,'model_call_attempted':False}
        if remaining_seconds <= 0:
            return None, {**event,'status':'timeout'}
        if len(text) > 1200:
            return None, {**event,'status':'input_too_long'}
        if text in self.cache:
            self.cache_hits += 1
            self.cache.move_to_end(text)
            candidate, reason = self.cache[text]
            return candidate, {**event,'status':'cache_hit','variant':candidate,'rejected_reason':reason}
        # Already well-formed requests can use the original directly.
        if not abbreviation_hints(text) and any(ord(c)>127 for c in text):
            return None, {**event,'status':'no_noise_hint'}
        self.load()
        deadline = started + min(remaining_seconds,self.settings.correction_timeout_seconds)
        if time.monotonic() >= deadline:
            return None, {**event,'status':'timeout'}
        event.update(model_call_attempted=True,model_call_started=time.monotonic())
        self.calls += 1
        try:
            raw, reason = self._generate(text,deadline)
        except RuntimeError as exc:
            raise RetrievalFatalError(f'Correction GPU failure: {exc}') from exc
        if reason == 'input_too_long':
            self.calls -= 1
            event['model_call_attempted'] = False
            event.pop('model_call_started',None)
        else:
            event['raw'] = raw
        if raw is not None:
            reason = validate_variant(text,raw)
        candidate = raw if raw and not reason and raw != text else None
        if reason not in {'timeout','incomplete_output'}:
            self.cache[text] = (candidate,reason)
            while len(self.cache)>128:
                self.cache.popitem(last=False)
        return candidate, {**event,'status':'accepted' if candidate else reason or 'unchanged',
            'variant':candidate,'rejected_reason':reason,'seconds':time.monotonic()-started}

    def close(self):
        self.model = self.tokenizer = None
        self.cache.clear()
