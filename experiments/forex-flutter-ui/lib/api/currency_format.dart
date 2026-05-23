// Centralised currency-symbol lookup so the dashboard, top bar,
// trade-watch screen, and growth-mode card all render the right
// glyph regardless of which currency the broker reports.
//
// Task #145 / #94: every previous call site did
//     `currency == 'EUR' ? '€' : r'$'`
// which renders CHF / GBP / JPY accounts as `$` — wrong and confusing
// for non-USD operators. Per task #94 there should be NO fallback
// hardcoded values; on unknown ISO codes we fall back to the ISO
// string itself so the UI never silently lies.

/// Returns the typographic glyph for a 3-letter ISO currency code.
/// Falls back to the ISO code string (NOT a `$` literal) when the
/// code isn't in the table. Case-insensitive on the input.
String currencyGlyph(String isoCode) {
  switch (isoCode.toUpperCase()) {
    case 'USD':
      return r'$';
    case 'EUR':
      return '€';
    case 'GBP':
      return '£';
    case 'JPY':
      return '¥';
    case 'CHF':
      return 'Fr';
    case 'AUD':
      return r'A$';
    case 'NZD':
      return r'NZ$';
    case 'CAD':
      return r'C$';
    case 'PLN':
      return 'zł';
    default:
      // Strict per #94: don't fabricate a symbol. Show the ISO code so
      // the operator at least knows which currency the broker said
      // their account is in, even if our table doesn't have a glyph
      // for it yet.
      return isoCode.toUpperCase();
  }
}
