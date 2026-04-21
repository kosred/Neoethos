#[cfg(target_os = "windows")]
use tracing::info;
use tracing::warn;

#[cfg(target_os = "windows")]
use std::ffi::OsString;
#[cfg(target_os = "windows")]
use std::os::windows::ffi::OsStringExt;
#[cfg(target_os = "windows")]
use windows::Win32::Foundation::{HWND, LPARAM};
#[cfg(target_os = "windows")]
use windows::Win32::UI::Input::KeyboardAndMouse::{
    INPUT, INPUT_KEYBOARD, KEYBDINPUT, KEYEVENTF_KEYUP, SendInput, VIRTUAL_KEY, VK_CONTROL, VK_E,
};
#[cfg(target_os = "windows")]
use windows::Win32::UI::WindowsAndMessaging::{
    EnumWindows, GetWindowTextLengthW, GetWindowTextW, IsIconic, SW_RESTORE, SetForegroundWindow,
    ShowWindow,
};
#[cfg(target_os = "windows")]
use windows::core::BOOL;

pub fn ensure_autotrading_enabled() -> bool {
    #[cfg(not(target_os = "windows"))]
    {
        return true;
    }

    #[cfg(target_os = "windows")]
    {
        warn!(
            "Cannot verify MT5 AutoTrading from forex-core without broker terminal state; use the MT5 adapter before live execution."
        );
        false
    }
}

#[cfg(not(target_os = "windows"))]
pub fn ensure_autotrading_window_shortcut() -> bool {
    true
}

#[cfg(target_os = "windows")]
pub fn ensure_autotrading_window_shortcut() -> bool {
    if !focus_mt5_window() {
        return false;
    }
    send_ctrl_e();
    true
}

#[cfg(target_os = "windows")]
pub fn focus_mt5_window() -> bool {
    unsafe {
        let mut found_hwnd: Option<HWND> = None;

        unsafe extern "system" fn enum_window_proc(hwnd: HWND, lparam: LPARAM) -> BOOL {
            unsafe {
                let found_ptr = lparam.0 as *mut Option<HWND>;

                let length = GetWindowTextLengthW(hwnd);
                if length > 0 {
                    let mut buffer = vec![0u16; (length + 1) as usize];
                    GetWindowTextW(hwnd, &mut buffer);
                    let title = OsString::from_wide(&buffer[..length as usize]);
                    let title_lossy = title.to_string_lossy();

                    if title_lossy.contains("MetaTrader 5") {
                        // Found it
                        *found_ptr = Some(hwnd);
                        return BOOL(0); // Stop enumeration
                    }
                }
                BOOL(1) // Continue enumeration
            }
        }

        let lparam = LPARAM(&mut found_hwnd as *mut _ as isize);
        let _ = EnumWindows(Some(enum_window_proc), lparam);

        if let Some(hwnd) = found_hwnd {
            info!("Found MT5 window. Focusing...");
            if IsIconic(hwnd).as_bool() {
                let _ = ShowWindow(hwnd, SW_RESTORE);
            }
            let _ = SetForegroundWindow(hwnd);
            return true;
        }
    }
    warn!("MetaTrader 5 window not found.");
    false
}

#[cfg(not(target_os = "windows"))]
pub fn focus_mt5_window() -> bool {
    warn!("Window control is not supported on this OS.");
    false
}

#[cfg(target_os = "windows")]
pub fn send_ctrl_e() {
    unsafe {
        let inputs = [
            input_key(VK_CONTROL, false),
            input_key(VK_E, false),
            input_key(VK_E, true),
            input_key(VK_CONTROL, true),
        ];

        SendInput(&inputs, std::mem::size_of::<INPUT>() as i32);
    }
}

#[cfg(not(target_os = "windows"))]
pub fn send_ctrl_e() {
    warn!("Keyboard input is not supported on this OS.");
}

#[cfg(target_os = "windows")]
fn input_key(vk: VIRTUAL_KEY, up: bool) -> INPUT {
    let mut input = INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: Default::default(),
    };

    let flags = if up {
        KEYEVENTF_KEYUP
    } else {
        Default::default()
    };

    input.Anonymous.ki = KEYBDINPUT {
        wVk: vk,
        wScan: 0,
        dwFlags: flags,
        time: 0,
        dwExtraInfo: 0,
    };

    input
}
