//+------------------------------------------------------------------+
//| NeoEthosSignalReader.mq5                                        |
//|                                                                  |
//| Reads the signal.json written by:                               |
//|   neoethos-cli live-signal --portfolio <path> --output <file>   |
//| and places/closes orders on the attached chart symbol.          |
//|                                                                  |
//| Setup:                                                           |
//| 1. Copy this file to MT5 > File > Open Data Folder >            |
//|    MQL5\Experts\                                                 |
//| 2. In MetaEditor press F7 to compile.                           |
//| 3. Attach to a chart. Set SignalFile input to the output path   |
//|    of neoethos-cli live-signal (must be inside MT5 Common\Files |
//|    or an absolute path the OS allows MT5 to read).              |
//| 4. Schedule neoethos-cli live-signal via Windows Task Scheduler |
//|    to run once per base timeframe (e.g. every hour for H1).     |
//+------------------------------------------------------------------+
#property copyright "NeoEthos"
#property version   "1.10"
#property description "Reads NeoEthos gene signal and trades accordingly"

#include <Trade\Trade.mqh>

//--- Inputs
input string SignalFile      = "neoethos_signal.json"; // Signal file name (in MT5 Common\Files)
input double LotSize         = 0.01;                   // Position size (lots)
input double StopLossPips    = 20.0;                   // Stop-loss (pips; 0 = no SL)
input double TakeProfitPips  = 40.0;                   // Take-profit (pips; 0 = no TP)
input int    MagicNumber     = 20260620;               // Magic number (unique per EA instance)
input bool   PrintSignals    = true;                   // Print signal details to Experts log

//--- Globals
CTrade   g_trade;
long     g_last_bar_ts  = 0;   // bar_ts_ms from the last signal we acted on
string   g_last_signal  = "";  // last direction: "Long" / "Short" / "Flat"
datetime g_last_bar_open = 0;  // MT5 bar open time — prevents re-firing on the same bar

//+------------------------------------------------------------------+
int OnInit()
{
    g_trade.SetExpertMagicNumber(MagicNumber);
    g_trade.SetDeviationInPoints(20);   // 2-pip slippage tolerance on market fills
    Print("NeoEthos EA ready. SignalFile=", SignalFile,
          " Symbol=", _Symbol, " TF=", EnumToString(Period()));
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason) {}

//+------------------------------------------------------------------+
void OnTick()
{
    // Fire only once per bar (on the first tick of a new bar)
    datetime bar_open = iTime(_Symbol, PERIOD_CURRENT, 0);
    if (bar_open == g_last_bar_open)
        return;
    g_last_bar_open = bar_open;

    string json = ReadSignalFile();
    if (json == "")
        return;

    ProcessSignal(json);
}

//+------------------------------------------------------------------+
string ReadSignalFile()
{
    // FILE_COMMON reads from %AppData%\MetaQuotes\Terminal\Common\Files\
    int fh = FileOpen(SignalFile, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON);
    if (fh == INVALID_HANDLE)
    {
        if (PrintSignals)
            Print("NeoEthos EA: cannot open '", SignalFile,
                  "' in Common\\Files — run neoethos-cli live-signal with "
                  "--output pointing to that folder.");
        return "";
    }
    string content = "";
    while (!FileIsEnding(fh))
        content += FileReadString(fh);
    FileClose(fh);
    return content;
}

//+------------------------------------------------------------------+
void ProcessSignal(const string &json)
{
    // Check freshness: only act on bars we haven't seen
    long ts = ParseLong(json, "bar_ts_ms");
    if (ts > 0 && ts <= g_last_bar_ts)
    {
        if (PrintSignals)
            Print("NeoEthos EA: signal bar_ts_ms=", ts, " already processed — skipping");
        return;
    }

    string signal    = ParseString(json, "signal");   // "Long" | "Short" | "Flat"
    string sym       = ParseString(json, "symbol");
    string base_tf   = ParseString(json, "base_tf");
    double conf      = ParseDouble(json, "confidence");

    if (signal == "")
    {
        Print("NeoEthos EA: could not parse 'signal' field from JSON");
        return;
    }

    if (ts > 0)
        g_last_bar_ts = ts;

    if (PrintSignals)
        Print("NeoEthos EA: signal=", signal,
              " sym=", sym, " tf=", base_tf,
              " conf=", DoubleToString(conf, 3),
              " bar_ts=", ts);

    if (signal == g_last_signal)
    {
        if (PrintSignals)
            Print("NeoEthos EA: no direction change (", signal, ") — holding");
        return;
    }
    g_last_signal = signal;

    if (signal == "Long")
    {
        CloseAll(POSITION_TYPE_SELL);
        if (CountPositions(POSITION_TYPE_BUY) == 0)
            OpenBuy();
    }
    else if (signal == "Short")
    {
        CloseAll(POSITION_TYPE_BUY);
        if (CountPositions(POSITION_TYPE_SELL) == 0)
            OpenSell();
    }
    else // "Flat" or unknown
    {
        CloseAll(-1); // close all directions
    }
}

//+------------------------------------------------------------------+
void CloseAll(int type)
{
    for (int i = PositionsTotal() - 1; i >= 0; i--)
    {
        ulong ticket = PositionGetTicket(i);
        if (ticket == 0) continue;
        if (PositionGetString(POSITION_SYMBOL)  != _Symbol)     continue;
        if (PositionGetInteger(POSITION_MAGIC)  != MagicNumber) continue;
        if (type != -1 && (int)PositionGetInteger(POSITION_TYPE) != type) continue;
        if (!g_trade.PositionClose(ticket))
            Print("NeoEthos EA: close failed ticket=", ticket,
                  " err=", GetLastError());
    }
}

//+------------------------------------------------------------------+
int CountPositions(int type)
{
    int count = 0;
    for (int i = 0; i < PositionsTotal(); i++)
    {
        ulong ticket = PositionGetTicket(i);
        if (ticket == 0) continue;
        if (PositionGetString(POSITION_SYMBOL)  != _Symbol)     continue;
        if (PositionGetInteger(POSITION_MAGIC)  != MagicNumber) continue;
        if ((int)PositionGetInteger(POSITION_TYPE) == type)     count++;
    }
    return count;
}

//+------------------------------------------------------------------+
void OpenBuy()
{
    double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double pip   = PipSize();
    double sl    = StopLossPips   > 0 ? NormalizeDouble(price - StopLossPips   * pip, _Digits) : 0;
    double tp    = TakeProfitPips > 0 ? NormalizeDouble(price + TakeProfitPips * pip, _Digits) : 0;
    if (!g_trade.Buy(LotSize, _Symbol, price, sl, tp, "NeoEthos-Auto"))
        Print("NeoEthos EA: BUY failed err=", GetLastError());
    else
        Print("NeoEthos EA: BUY ", LotSize, " ", _Symbol,
              " @", DoubleToString(price, _Digits),
              " SL=", DoubleToString(sl, _Digits),
              " TP=", DoubleToString(tp, _Digits));
}

//+------------------------------------------------------------------+
void OpenSell()
{
    double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double pip   = PipSize();
    double sl    = StopLossPips   > 0 ? NormalizeDouble(price + StopLossPips   * pip, _Digits) : 0;
    double tp    = TakeProfitPips > 0 ? NormalizeDouble(price - TakeProfitPips * pip, _Digits) : 0;
    if (!g_trade.Sell(LotSize, _Symbol, price, sl, tp, "NeoEthos-Auto"))
        Print("NeoEthos EA: SELL failed err=", GetLastError());
    else
        Print("NeoEthos EA: SELL ", LotSize, " ", _Symbol,
              " @", DoubleToString(price, _Digits),
              " SL=", DoubleToString(sl, _Digits),
              " TP=", DoubleToString(tp, _Digits));
}

//+------------------------------------------------------------------+
double PipSize()
{
    double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
    // 5-digit / 3-digit brokers: 1 pip = 10 points
    return point * ((_Digits == 5 || _Digits == 3) ? 10.0 : 1.0);
}

//+------------------------------------------------------------------+
// Minimal JSON field extractors (no external library needed)
//+------------------------------------------------------------------+
long ParseLong(const string &json, const string &key)
{
    string needle = "\"" + key + "\":";
    int pos = StringFind(json, needle);
    if (pos < 0) return 0;
    pos += StringLen(needle);
    while (pos < StringLen(json) && StringGetCharacter(json, pos) == ' ') pos++;
    string num = "";
    ushort c;
    while (pos < StringLen(json) && (c = StringGetCharacter(json, pos)) >= '0' && c <= '9')
    { num += ShortToString(c); pos++; }
    return (num == "") ? 0 : StringToInteger(num);
}

//+------------------------------------------------------------------+
string ParseString(const string &json, const string &key)
{
    string needle = "\"" + key + "\":";
    int pos = StringFind(json, needle);
    if (pos < 0) return "";
    pos += StringLen(needle);
    while (pos < StringLen(json) && StringGetCharacter(json, pos) != '"') pos++;
    pos++; // skip opening quote
    string val = "";
    ushort c;
    while (pos < StringLen(json) && (c = StringGetCharacter(json, pos)) != '"')
    { val += ShortToString(c); pos++; }
    return val;
}

//+------------------------------------------------------------------+
double ParseDouble(const string &json, const string &key)
{
    string needle = "\"" + key + "\":";
    int pos = StringFind(json, needle);
    if (pos < 0) return 0.0;
    pos += StringLen(needle);
    while (pos < StringLen(json) && StringGetCharacter(json, pos) == ' ') pos++;
    string num = "";
    ushort c;
    while (pos < StringLen(json) &&
           ((c = StringGetCharacter(json, pos)) >= '0' && c <= '9' || c == '.' || c == '-'))
    { num += ShortToString(c); pos++; }
    return (num == "") ? 0.0 : StringToDouble(num);
}
