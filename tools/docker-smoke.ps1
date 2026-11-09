param([string]$Image = 'datamate-dev:local')
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
Set-Location -LiteralPath $projectRoot
$suffix = [guid]::NewGuid().ToString('N').Substring(0, 10)
$stageName = "datamate-prep-$suffix"
$runName = "datamate-run-$suffix"
$dataVolume = "datamate-data-$suffix"
$resultVolume = "datamate-result-$suffix"
$inputDir = Join-Path $projectRoot "outputs/runs/docker-smoke/input-$suffix"
$outputDir = Join-Path $projectRoot "outputs/runs/docker-smoke/result-$suffix"
$sourceRoot = Join-Path $projectRoot 'train/train'
New-Item -ItemType Directory -Force $inputDir, $outputDir | Out-Null
$questionLines = @(Get-Content -LiteralPath (Join-Path $sourceRoot 'questions.jsonl') -Encoding utf8 -TotalCount 3)
$questions = @($questionLines | ForEach-Object { $_ | ConvertFrom-Json })
if ($questions.Count -ne 3) { throw 'Cần tối thiểu 3 câu train' }
$utf8 = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllText((Join-Path $inputDir 'questions.jsonl'), ([string]::Join([char]10, $questionLines) + [char]10), $utf8)
foreach ($file in @('rulebook.md', 'calendar.csv')) {
    Copy-Item -LiteralPath (Join-Path $sourceRoot $file) -Destination (Join-Path $inputDir $file)
}
New-Item -ItemType Directory -Force (Join-Path $inputDir 'data') | Out-Null
foreach ($business in @($questions.business_id | Sort-Object -Unique)) {
    if ($business -notmatch '^[A-Za-z0-9_-]+$') { throw 'Business ID không hợp lệ' }
    $businessSource = (Resolve-Path -LiteralPath (Join-Path $sourceRoot "data/$business")).Path
    $dataRoot = (Resolve-Path -LiteralPath (Join-Path $sourceRoot 'data')).Path
    if ((Split-Path -Parent $businessSource) -ne $dataRoot) { throw 'Đường dẫn nguồn vượt thư mục data' }
    Copy-Item -LiteralPath $businessSource -Destination (Join-Path $inputDir "data/$business") -Recurse
}
try {
    docker volume create --label "datamate.smoke=$suffix" $dataVolume | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Không tạo được data volume' }
    docker volume create --label "datamate.smoke=$suffix" $resultVolume | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Không tạo được result volume' }
    docker create --name $stageName --label "datamate.smoke=$suffix" --network none --mount "type=volume,source=$dataVolume,target=/data" --entrypoint /bin/true $Image | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Không tạo được staging container' }
    docker cp "$inputDir/." "${stageName}:/data"
    if ($LASTEXITCODE -ne 0) { throw 'Không chép được đúng subset vào data volume' }
    docker create --name $runName --label "datamate.smoke=$suffix" --network none --read-only --tmpfs /work:rw,size=512m --tmpfs /tmp:rw,size=128m --mount "type=volume,source=$dataVolume,target=/data,readonly" --mount "type=volume,source=$resultVolume,target=/result" $Image --limit 3 | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Không tạo được inference container' }
    docker start -a $runName
    if ($LASTEXITCODE -ne 0) { throw 'Inference container không hoàn tất' }
    $metadata = @(docker inspect $runName | ConvertFrom-Json)[0]
    if ($metadata.State.ExitCode -ne 0) { throw "Exit code: $($metadata.State.ExitCode)" }
    $dataMount = @($metadata.Mounts | Where-Object { $_.Destination -eq '/data' })[0]
    if ($metadata.HostConfig.NetworkMode -ne 'none' -or -not $metadata.HostConfig.ReadonlyRootfs -or $dataMount.RW) {
        throw 'Container không đáp ứng điều kiện offline/chỉ đọc'
    }
    docker cp "${runName}:/result/." "$outputDir"
    if ($LASTEXITCODE -ne 0) { throw 'Không chép được kết quả về E:' }
    $submission = @(Get-Content -LiteralPath (Join-Path $outputDir 'submission.jsonl') -Encoding utf8 | ForEach-Object { $_ | ConvertFrom-Json })
    $expected = @($questions.id | Sort-Object)
    $actual = @($submission.id | Sort-Object)
    if ($submission.Count -ne 3 -or (Compare-Object $expected $actual)) { throw 'Submission thiếu/trùng/sai ID' }
    if (@($submission | Where-Object { $_.type -ne 'abstain' }).Count -ne 0) { throw 'Smoke demo phải trả abstain' }
    $logFiles = @(Get-ChildItem -LiteralPath (Join-Path $outputDir 'logs') -Filter '*.json' -File)
    if ($logFiles.Count -ne 3) { throw 'Thiếu log từng câu' }
    $verification = @{ image=$Image; image_id=$metadata.Image; network=$metadata.HostConfig.NetworkMode;
        rootfs_readonly=$metadata.HostConfig.ReadonlyRootfs; data_readonly=(-not $dataMount.RW);
        exit_code=$metadata.State.ExitCode; questions=3; answer_types=@($submission.type);
        note='Demo packaging check, not an LLM benchmark' }
    $verification | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $outputDir 'docker-verification.json') -Encoding utf8
    Write-Output "Smoke test OK. Kết quả: $outputDir"
} finally {
    # Remove only resources carrying this invocation's unique ownership label.
    foreach ($containerName in @($runName, $stageName)) {
        $record = @(docker inspect $containerName 2>$null | ConvertFrom-Json)[0]
        if ($record -and $record.Config.Labels.'datamate.smoke' -eq $suffix) { docker rm -f $containerName | Out-Null }
    }
    foreach ($volumeName in @($dataVolume, $resultVolume)) {
        $record = @(docker volume inspect $volumeName 2>$null | ConvertFrom-Json)[0]
        if ($record -and $record.Labels.'datamate.smoke' -eq $suffix) { docker volume rm $volumeName | Out-Null }
    }
}
