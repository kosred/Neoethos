import 'dart:io';
import 'lib/startup/backend_supervisor.dart' as bs;
void main() {
  // The merger is private — exercise it indirectly by calling
  // _seedUserDataDir through a constructed supervisor instance.
  // For pure-dart smoke we re-implement minimal trace:
  final user = File('${Platform.environment["LOCALAPPDATA"]}\\neoethos\\config.yaml').readAsStringSync();
  final bundle = File('C:\\Program Files\\NeoEthos\\config.yaml').readAsStringSync();
  print('USER LEN: ${user.length}');
  print('BUNDLE LEN: ${bundle.length}');
  print('user has account_currency? ${user.contains("account_currency")}');
  print('bundle has account_currency? ${bundle.contains("account_currency")}');
}