# run-signal.ps1
# Τρέχει από Windows Task Scheduler κάθε X λεπτά (ανάλογα το TF).
# Γράφει το signal στο MT5 Common\Files για τον EA.
#
# ΧΡΗΣΗ:
#   Μη αυτόματα: powershell -File "C:\Users\konst\development\forex-ai\mt5\run-signal.ps1"
#   Task Scheduler: Action = powershell.exe -NonInteractive -File "<αυτό_το_αρχείο>"
#                   Trigger = "Every 1 hour" (για H1)  /  "Every 15 min" (για M15)

$CLI      = "C:\Users\konst\development\forex-ai\target\release\neoethos-cli.exe"
$DATA     = "C:\Users\konst\development\forex-ai\data"
$CACHE    = "C:\Users\konst\development\forex-ai\cache"
$OUTDIR   = "C:\Users\konst\AppData\Roaming\MetaQuotes\Terminal\Common\Files"
$OUTFILE  = "$OUTDIR\neoethos_signal.json"

# Τα portfolio paths — τροποποίησε ανάλογα τι θέλεις να τρέχει
$PORTFOLIOS = @(
    "$CACHE\auto_loop_propfirm\EURUSD_H1.json.live_portfolio.json",
    "$CACHE\auto_loop_propfirm\GBPUSD_H1.json.live_portfolio.json",
    "$CACHE\auto_loop_propfirm\EURUSD_H4.json.live_portfolio.json"
)

# Τρέχει για το πρώτο portfolio που υπάρχει
foreach ($portfolio in $PORTFOLIOS) {
    if (Test-Path $portfolio) {
        Write-Host "[$(Get-Date -f 'HH:mm:ss')] Running live-signal for $portfolio"
        & $CLI live-signal `
            --portfolio $portfolio `
            --root $DATA `
            --output $OUTFILE
        Write-Host "Signal written to $OUTFILE"
        break
    }
}
