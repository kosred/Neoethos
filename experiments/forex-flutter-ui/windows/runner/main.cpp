#include <flutter/dart_project.h>
#include <flutter/flutter_view_controller.h>
#include <windows.h>

#include "flutter_window.h"
#include "utils.h"

int APIENTRY wWinMain(_In_ HINSTANCE instance, _In_opt_ HINSTANCE prev,
                      _In_ wchar_t *command_line, _In_ int show_command) {
  // Attach to console when present (e.g., 'flutter run') or create a
  // new console when running with a debugger.
  if (!::AttachConsole(ATTACH_PARENT_PROCESS) && ::IsDebuggerPresent()) {
    CreateAndAttachConsole();
  }

  // Initialize COM, so that it is available for use in the library and/or
  // plugins.
  ::CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);

  // **Task #266 fix (2026-05-26)**: defensive DLL-search-path setup.
  //
  // Symptom: csrss-hosted Win32 dialog "Code execution cannot continue
  // because url_launcher_windows_plugin.dll was not found" pops up
  // intermittently on a running NeoEthos. The DLL is PRESENT next to
  // NeoEthos.exe — Windows' default loader normally finds it via the
  // "application directory" search step. But on certain launch paths
  // (Start Menu shortcut → ShellExecute with a non-canonical CWD,
  // ProcessExplorer-style indirect spawn, NSIS uninstall handle still
  // open) the application-directory step is skipped on the loader's
  // delayed-resolution path for plugin DLLs.
  //
  // Belt-and-braces: explicitly pin the DLL search to the .exe's
  // directory via `SetDllDirectoryW`. This is the legacy but
  // rock-solid API supported by every Windows version we target.
  // No-op when the default search path already includes us, defensive
  // safety net when it doesn't.
  {
    wchar_t exe_path[MAX_PATH];
    DWORD len = ::GetModuleFileNameW(nullptr, exe_path, MAX_PATH);
    if (len > 0 && len < MAX_PATH) {
      // Strip filename component → leave directory.
      for (DWORD i = len; i-- > 0;) {
        if (exe_path[i] == L'\\' || exe_path[i] == L'/') {
          exe_path[i] = L'\0';
          break;
        }
      }
      ::SetDllDirectoryW(exe_path);
    }
  }

  flutter::DartProject project(L"data");

  std::vector<std::string> command_line_arguments =
      GetCommandLineArguments();

  project.set_dart_entrypoint_arguments(std::move(command_line_arguments));

  FlutterWindow window(project);
  Win32Window::Point origin(10, 10);
  // 1400×900 matches the mockup's intended dock density (4 stat cards
  // across + 3-column engine-health row + sidebar). The previous
  // 1280×720 was the Flutter default; on Windows DPI scaling that
  // collapsed the dashboard ribbon and forced horizontal scrolling
  // in the top bar.
  Win32Window::Size size(1400, 900);
  // Window title shown in the OS chrome (Alt-Tab, taskbar, title bar).
  // Keep in lock-step with the brand string in `lib/widgets/topbar.dart`.
  if (!window.Create(L"NeoEthos", origin, size)) {
    return EXIT_FAILURE;
  }
  window.SetQuitOnClose(true);

  ::MSG msg;
  while (::GetMessage(&msg, nullptr, 0, 0)) {
    ::TranslateMessage(&msg);
    ::DispatchMessage(&msg);
  }

  ::CoUninitialize();
  return EXIT_SUCCESS;
}
