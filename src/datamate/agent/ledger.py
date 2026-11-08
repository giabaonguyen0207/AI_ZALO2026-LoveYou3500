"""Guarded cash-flow calculations driven by the current split's debt policy."""
from ..domain import DataMateError


def debt_flow(plan,tools):
    if not tools.policy.debt_flow_supported:raise DataMateError('Rulebook chưa xác nhận luật nợ thêm/trả sau ngày mua')
    caps=tools.semantic['capabilities']
    if not caps.get('payments') or not caps.get('order_total'):
        raise DataMateError('Nợ thêm/thu nợ cần tổng phải trả và sổ thu đã ánh xạ; snapshot dư nợ không thay thế dòng tiền')
    bad=tools.executor.execute("SELECT COUNT(*) FROM dm_orders_all WHERE order_date IS NULL OR status='unknown' OR payment_method IS NULL OR payment_method='unknown' OR (payment_method='debt' AND order_total IS NULL)")['rows'][0][0]
    bad+=tools.executor.execute('SELECT COUNT(*) FROM dm_payments WHERE payment_date IS NULL OR amount IS NULL OR amount<0')['rows'][0][0]
    orphan=tools.executor.execute('SELECT COUNT(*) FROM dm_payments p LEFT JOIN dm_orders_all o ON o.order_id=p.order_id WHERE o.order_id IS NULL')['rows'][0][0]
    if bad or orphan:raise DataMateError('Sổ thu/đơn chưa đủ ngày, tiền, khóa, trạng thái hoặc hình thức thanh toán; không cộng dòng tiền một phần')
    mismatch=tools.executor.execute('SELECT COUNT(*) FROM dm_orders_all o LEFT JOIN (SELECT order_id,SUM(amount) AS total FROM dm_payments GROUP BY order_id) p ON o.order_id=p.order_id WHERE o.paid IS NOT NULL AND ABS(o.paid-COALESCE(p.total,0))>0.01')['rows'][0][0]
    if mismatch:raise DataMateError('Sổ thu chưa đối soát với tổng đã trả; không tính dòng tiền trên ledger thiếu')
    start,end=(plan['resolved_periods'][0][k] for k in ('start','end'))
    # Caller only permits customer identity predicates; no ignored arbitrary filters.
    customer=next((p['value'] for p in plan['filters'] if p['column']=='customer_key'),None)
    where="o.status!='cancelled' AND o.payment_method='debt'";extra=[]
    if customer is not None:where+=' AND o.customer_key=?';extra=[customer]
    added=("SELECT COALESCE(SUM(MAX(o.order_total-COALESCE(p.at_purchase,0),0)),0) AS value "
           "FROM dm_orders_all o LEFT JOIN (SELECT p.order_id,SUM(p.amount) AS at_purchase FROM dm_payments p "
           "JOIN dm_orders_all x ON x.order_id=p.order_id WHERE p.payment_date=x.order_date GROUP BY p.order_id) p "
           "ON p.order_id=o.order_id WHERE o.order_date BETWEEN ? AND ? AND "+where)
    paid=("SELECT COALESCE(SUM(p.amount),0) AS value FROM dm_payments p JOIN dm_orders_all o ON p.order_id=o.order_id "
          "WHERE p.payment_date BETWEEN ? AND ? AND p.payment_date>o.order_date AND "+where)
    if plan['flow']=='both':
        sql='SELECT value FROM (SELECT 0 AS ordinal,value FROM ('+added+') UNION ALL SELECT 1,value FROM ('+paid+')) ORDER BY ordinal'
        params=[start,end,*extra,start,end,*extra]
    else:sql=added if plan['flow']=='added' else paid;params=[start,end,*extra]
    result=tools.executor.execute(sql,params)
    return {**result,'verified':True,'availability':'available','answer_type':plan['type'],
            'verification_scope':'debt_flow_rulebook_and_reconciled_payment_ledger','plan':plan}
