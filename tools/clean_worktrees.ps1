<#
.SYNOPSIS
Убрать рабочие копии агентов (git worktree), не задев данные основного репозитория.

.DESCRIPTION
У рабочих копий data/ и runs/ подключены junction-ссылками на основной
репозиторий. git worktree remove и рекурсивное удаление идут по ссылке внутрь
и стирают цель (docs/pitfalls.md, 13.1). Поэтому порядок такой:

1. снять все ссылки (junction, symlink) в рабочей копии, не заходя в них:
   [IO.Directory]::Delete(путь, $false) удаляет только саму ссылку;
2. проверить, что точек повторной обработки не осталось — иначе отказ;
3. git worktree remove --force, остаток каталога — удалить, git worktree prune.

.PARAMETER Path
Рабочие копии для уборки. Без него — все, кроме основной.

.EXAMPLE
tools\clean_worktrees.ps1 -WhatIf
tools\clean_worktrees.ps1 -Path ..\lol-wt-788
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [string[]]$Path
)

$ErrorActionPreference = 'Stop'

function Get-Worktrees {
    # Первая запись porcelain — основная рабочая копия.
    $list = git worktree list --porcelain | Where-Object { $_ -like 'worktree *' } |
        ForEach-Object { [IO.Path]::GetFullPath(($_ -replace '^worktree ', '')) }
    if ($LASTEXITCODE -ne 0) { throw 'git worktree list не сработал — запускать из репозитория' }
    return , @($list)
}

function Find-Links {
    # Обход без захода в ссылки: Get-ChildItem -Recurse в Windows PowerShell
    # ходит по junction, а этого здесь и надо избежать.
    param([string]$Root)
    $out = New-Object System.Collections.Generic.List[string]
    $stack = New-Object System.Collections.Generic.Stack[string]
    $stack.Push($Root)
    while ($stack.Count) {
        $dir = $stack.Pop()
        foreach ($e in [IO.Directory]::EnumerateFileSystemEntries($dir)) {
            $attr = [IO.File]::GetAttributes($e)
            if ($attr -band [IO.FileAttributes]::ReparsePoint) { $out.Add($e) }
            elseif ($attr -band [IO.FileAttributes]::Directory) { $stack.Push($e) }
        }
    }
    return , $out
}

$all = Get-Worktrees
$main = $all[0]
$targets = if ($Path) { $Path | ForEach-Object { [IO.Path]::GetFullPath((Resolve-Path -LiteralPath $_).Path) } }
           else { $all | Select-Object -Skip 1 }

foreach ($wt in $targets) {
    if ($wt.TrimEnd('\') -ieq $main.TrimEnd('\')) { throw "$wt — основная рабочая копия, её не трогаю" }
    if ($all -notcontains $wt) { throw "$wt — не рабочая копия этого репозитория" }
    if (-not (Test-Path -LiteralPath $wt)) {
        Write-Output "$wt — каталога нет, только prune"
        continue
    }

    $links = Find-Links $wt
    foreach ($l in $links) {
        if ($PSCmdlet.ShouldProcess($l, 'снять ссылку')) {
            if ([IO.File]::GetAttributes($l) -band [IO.FileAttributes]::Directory) {
                [IO.Directory]::Delete($l, $false)
            } else {
                [IO.File]::Delete($l)
            }
            Write-Output "  ссылка снята: $l"
        }
    }
    if ($WhatIfPreference) { continue }

    $left = Find-Links $wt
    if ($left.Count) { throw "в $wt остались ссылки, удаление отменено:`n  $($left -join "`n  ")" }

    if ($PSCmdlet.ShouldProcess($wt, 'git worktree remove')) {
        git worktree remove --force $wt
        if (Test-Path -LiteralPath $wt) { Remove-Item -LiteralPath $wt -Recurse -Force }
        Write-Output "убрана: $wt"
    }
}
if (-not $WhatIfPreference) { git worktree prune }
