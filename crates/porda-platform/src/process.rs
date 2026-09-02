use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct ProcessInfo {
    pub name: String,
    pub pid: u32,
}

pub fn get_foreground_window() -> Option<WindowHandle> {
    #[cfg(target_os = "windows")]
    {
        windows_impl::get_foreground_window()
    }
    #[cfg(not(target_os = "windows"))]
    {
        tracing::warn!("get_foreground_window not supported on this platform");
        None
    }
}

pub fn get_window_rect(_hwnd: WindowHandle) -> Option<porda_vision::geometry::ScreenRect> {
    #[cfg(target_os = "windows")]
    {
        windows_impl::get_window_rect(_hwnd)
    }
    #[cfg(not(target_os = "windows"))]
    {
        None
    }
}

pub fn get_client_rect(_hwnd: WindowHandle) -> Option<porda_vision::geometry::ScreenRect> {
    #[cfg(target_os = "windows")]
    {
        windows_impl::get_client_rect(_hwnd)
    }
    #[cfg(not(target_os = "windows"))]
    {
        None
    }
}

pub fn get_window_process_name(_hwnd: WindowHandle) -> Option<String> {
    #[cfg(target_os = "windows")]
    {
        windows_impl::get_window_process_name(_hwnd)
    }
    #[cfg(not(target_os = "windows"))]
    {
        None
    }
}

pub fn get_window_title(_hwnd: WindowHandle) -> Option<String> {
    #[cfg(target_os = "windows")]
    {
        windows_impl::get_window_title(_hwnd)
    }
    #[cfg(not(target_os = "windows"))]
    {
        None
    }
}

pub fn is_window_visible(_hwnd: WindowHandle) -> bool {
    #[cfg(target_os = "windows")]
    {
        windows_impl::is_window_visible(_hwnd)
    }
    #[cfg(not(target_os = "windows"))]
    {
        false
    }
}

pub fn is_window_minimized(_hwnd: WindowHandle) -> bool {
    #[cfg(target_os = "windows")]
    {
        windows_impl::is_window_minimized(_hwnd)
    }
    #[cfg(not(target_os = "windows"))]
    {
        false
    }
}

pub fn capture_window(_hwnd: WindowHandle) -> Option<porda_vision::detection::FrameData> {
    #[cfg(target_os = "windows")]
    {
        windows_impl::capture_window(_hwnd)
    }
    #[cfg(not(target_os = "windows"))]
    {
        None
    }
}

pub fn capture_screenshot() -> Option<porda_vision::detection::FrameData> {
    #[cfg(target_os = "windows")]
    {
        windows_impl::capture_screenshot()
    }
    #[cfg(not(target_os = "windows"))]
    {
        None
    }
}

pub fn set_window_topmost(_hwnd: WindowHandle) {
    #[cfg(target_os = "windows")]
    {
        windows_impl::set_window_topmost(_hwnd)
    }
}

pub fn set_process_realtime_priority() {
    #[cfg(target_os = "windows")]
    {
        windows_impl::set_process_realtime_priority()
    }
}

pub fn add_startup_registry() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(target_os = "windows")]
    {
        windows_impl::add_startup_registry()
    }
    #[cfg(not(target_os = "windows"))]
    {
        Err("Startup registry not supported on this platform".into())
    }
}

pub fn remove_startup_registry() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(target_os = "windows")]
    {
        windows_impl::remove_startup_registry()
    }
    #[cfg(not(target_os = "windows"))]
    {
        Err("Startup registry not supported on this platform".into())
    }
}

pub fn get_cpu_usage() -> f32 {
    #[cfg(target_os = "windows")]
    {
        windows_impl::get_cpu_usage()
    }
    #[cfg(not(target_os = "windows"))]
    {
        0.0
    }
}

pub fn show_message(title: &str, message: &str) {
    #[cfg(target_os = "windows")]
    {
        windows_impl::show_message(title, message)
    }
    #[cfg(not(target_os = "windows"))]
    {
        tracing::info!("Message: {} - {}", title, message);
    }
}

pub fn set_graphics_preference() {
    #[cfg(target_os = "windows")]
    {
        windows_impl::set_graphics_preference()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WindowHandle(pub usize);

impl WindowHandle {
    pub fn invalid() -> Self {
        Self(0)
    }

    pub fn is_valid(self) -> bool {
        self.0 != 0
    }
}

pub struct MonitorInfo {
    pub bounds: porda_vision::geometry::ScreenRect,
    pub work_area: porda_vision::geometry::ScreenRect,
    pub is_primary: bool,
}

pub fn get_monitors() -> Vec<MonitorInfo> {
    #[cfg(target_os = "windows")]
    {
        windows_impl::get_monitors()
    }
    #[cfg(not(target_os = "windows"))]
    {
        vec![MonitorInfo {
            bounds: porda_vision::geometry::ScreenRect::new(0, 0, 1920, 1080),
            work_area: porda_vision::geometry::ScreenRect::new(0, 0, 1920, 1080),
            is_primary: true,
        }]
    }
}

pub fn list_windows(
    _include: &[String],
    _exclude: &[String],
    _always_skip: &[(String, String)],
) -> Vec<(WindowHandle, String, String, porda_vision::geometry::ScreenRect)> {
    #[cfg(target_os = "windows")]
    {
        windows_impl::list_windows(_include, _exclude, _always_skip)
    }
    #[cfg(not(target_os = "windows"))]
    {
        vec![]
    }
}

pub fn check_duplicate_instances() -> bool {
    #[cfg(target_os = "windows")]
    {
        windows_impl::check_duplicate_instances()
    }
    #[cfg(not(target_os = "windows"))]
    {
        false
    }
}

pub fn ensure_app_directories() -> std::io::Result<()> {
    porda_config::defaults::ensure_directories()
}

pub fn model_paths() -> (PathBuf, PathBuf) {
    let external = porda_config::defaults::external_model_dir();
    let cfg_files: Vec<_> = std::fs::read_dir(&external)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "cfg")
                .unwrap_or(false)
        })
        .collect();

    let weights_files: Vec<_> = std::fs::read_dir(&external)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "weights")
                .unwrap_or(false)
        })
        .collect();

    if cfg_files.len() == 1 && weights_files.len() == 1 {
        (cfg_files[0].path(), weights_files[0].path())
    } else {
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.to_path_buf()))
            .unwrap_or_else(|| PathBuf::from("."));
        (
            exe_dir.join("model").join("pordav4x3.cfg"),
            exe_dir.join("model").join("porda-19200-lr-0005-909.weights"),
        )
    }
}

#[cfg(target_os = "windows")]
mod windows_impl {
    use super::*;
    use porda_vision::geometry::ScreenRect;

    pub fn get_foreground_window() -> Option<WindowHandle> {
        unsafe {
            let hwnd = windows::Win32::UI::WindowsAndMessaging::GetForegroundWindow();
            if hwnd.0.is_null() {
                None
            } else {
                Some(WindowHandle(hwnd.0 as usize))
            }
        }
    }

    pub fn get_window_rect(hwnd: WindowHandle) -> Option<ScreenRect> {
        unsafe {
            use windows::Win32::UI::WindowsAndMessaging::GetWindowRect;
            let handle = windows::Win32::Foundation::HWND(hwnd.0 as *mut _);
            let mut rect = windows::Win32::Foundation::RECT::default();
            if GetWindowRect(handle, &mut rect).is_ok() {
                Some(ScreenRect {
                    x: rect.left,
                    y: rect.top,
                    width: (rect.right - rect.left) as u32,
                    height: (rect.bottom - rect.top) as u32,
                })
            } else {
                None
            }
        }
    }

    pub fn get_client_rect(hwnd: WindowHandle) -> Option<ScreenRect> {
        unsafe {
            use windows::Win32::UI::WindowsAndMessaging::GetClientRect;
            use windows::Win32::Graphics::Gdi::ClientToScreen;
            let handle = windows::Win32::Foundation::HWND(hwnd.0 as *mut _);
            let mut rect = windows::Win32::Foundation::RECT::default();
            if GetClientRect(handle, &mut rect).is_ok() {
                let mut point = windows::Win32::Foundation::POINT { x: 0, y: 0 };
                let _ = ClientToScreen(handle, &mut point);
                Some(ScreenRect {
                    x: point.x,
                    y: point.y,
                    width: (rect.right - rect.left) as u32,
                    height: (rect.bottom - rect.top) as u32,
                })
            } else {
                None
            }
        }
    }

    pub fn get_window_process_name(hwnd: WindowHandle) -> Option<String> {
        unsafe {
            use windows::Win32::UI::WindowsAndMessaging::GetWindowThreadProcessId;
            use windows::Win32::System::Threading::OpenProcess;
            use windows::Win32::System::Threading::PROCESS_QUERY_LIMITED_INFORMATION;
            use windows::Win32::Foundation::CloseHandle;

            let handle = windows::Win32::Foundation::HWND(hwnd.0 as *mut _);
            let mut pid = 0u32;
            GetWindowThreadProcessId(handle, Some(&mut pid));

            let process_handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid).ok()?;
            let mut name = [0u16; 260];
            let mut size = 260u32;
            let result = windows::Win32::System::Threading::QueryFullProcessImageNameW(
                process_handle,
                windows::Win32::System::Threading::PROCESS_NAME_FORMAT(0),
                windows::core::PWSTR(name.as_mut_ptr()),
                &mut size,
            );
            let _ = CloseHandle(process_handle);

            if result.is_ok() {
                let name = String::from_utf16_lossy(&name[..size as usize]);
                std::path::Path::new(&name)
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
            } else {
                None
            }
        }
    }

    pub fn get_window_title(hwnd: WindowHandle) -> Option<String> {
        unsafe {
            use windows::Win32::UI::WindowsAndMessaging::GetWindowTextW;
            let handle = windows::Win32::Foundation::HWND(hwnd.0 as *mut _);
            let mut name = [0u16; 512];
            let len = GetWindowTextW(handle, &mut name);
            if len > 0 {
                Some(String::from_utf16_lossy(&name[..len as usize]))
            } else {
                None
            }
        }
    }

    pub fn is_window_visible(hwnd: WindowHandle) -> bool {
        unsafe {
            use windows::Win32::UI::WindowsAndMessaging::IsWindowVisible;
            let handle = windows::Win32::Foundation::HWND(hwnd.0 as *mut _);
            IsWindowVisible(handle).as_bool()
        }
    }

    pub fn is_window_minimized(hwnd: WindowHandle) -> bool {
        unsafe {
            use windows::Win32::UI::WindowsAndMessaging::IsIconic;
            let handle = windows::Win32::Foundation::HWND(hwnd.0 as *mut _);
            IsIconic(handle).as_bool()
        }
    }

    pub fn set_window_topmost(hwnd: WindowHandle) {
        unsafe {
            use windows::Win32::UI::WindowsAndMessaging::SetWindowPos;
            use windows::Win32::UI::WindowsAndMessaging::HWND_TOPMOST;
            use windows::Win32::UI::WindowsAndMessaging::SWP_NOMOVE;
            use windows::Win32::UI::WindowsAndMessaging::SWP_NOSIZE;
            let handle = windows::Win32::Foundation::HWND(hwnd.0 as *mut _);
            let _ = SetWindowPos(handle, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE);
        }
    }

    pub fn set_process_realtime_priority() {
        unsafe {
            use windows::Win32::System::Threading::SetPriorityClass;
            use windows::Win32::System::Threading::GetCurrentProcess;
            use windows::Win32::System::Threading::REALTIME_PRIORITY_CLASS;
            let handle = GetCurrentProcess();
            let _ = SetPriorityClass(handle, REALTIME_PRIORITY_CLASS);
        }
    }

    pub fn add_startup_registry() -> Result<(), Box<dyn std::error::Error>> {
        use windows::Win32::System::Registry::*;
        unsafe {
            let exe_path = std::env::current_exe()?;
            let value = format!("\"{}\" --startup_by_windows", exe_path.display());

            let mut key = HKEY_CURRENT_USER;
            let mut result_key = HKEY::default();
            let reg_path = windows::core::PCWSTR(
                windows::core::w!("Software\\Microsoft\\Windows\\CurrentVersion\\Run").as_ptr(),
            );

            RegOpenKeyExW(
                key,
                reg_path,
                0,
                KEY_SET_VALUE,
                &mut result_key,
            )?;

            let value_wide: Vec<u16> = value.encode_utf16().chain(std::iter::once(0)).collect();
            RegSetValueExW(
                result_key,
                windows::core::PCWSTR(windows::core::w!("PordaAi").as_ptr()),
                0,
                REG_SZ,
                value_wide.as_ptr() as _,
                (value_wide.len() * 2) as u32,
            )?;

            RegCloseKey(result_key)?;
            Ok(())
        }
    }

    pub fn remove_startup_registry() -> Result<(), Box<dyn std::error::Error>> {
        use windows::Win32::System::Registry::*;
        unsafe {
            let mut result_key = HKEY::default();
            let reg_path = windows::core::PCWSTR(
                windows::core::w!("Software\\Microsoft\\Windows\\CurrentVersion\\Run").as_ptr(),
            );

            RegOpenKeyExW(
                HKEY_CURRENT_USER,
                reg_path,
                0,
                KEY_SET_VALUE,
                &mut result_key,
            )?;

            RegDeleteValueW(
                result_key,
                windows::core::PCWSTR(windows::core::w!("PordaAi").as_ptr()),
            )?;

            RegCloseKey(result_key)?;
            Ok(())
        }
    }

    pub fn get_cpu_usage() -> f32 {
        0.0
    }

    pub fn show_message(title: &str, message: &str) {
        unsafe {
            use windows::Win32::UI::WindowsAndMessaging::MessageBoxW;
            let title_wide: Vec<u16> = title.encode_utf16().chain(std::iter::once(0)).collect();
            let msg_wide: Vec<u16> = message.encode_utf16().chain(std::iter::once(0)).collect();
            MessageBoxW(
                None,
                windows::core::PCWSTR(msg_wide.as_ptr()),
                windows::core::PCWSTR(title_wide.as_ptr()),
                windows::Win32::UI::WindowsAndMessaging::MB_OK,
            );
        }
    }

    pub fn set_graphics_preference() {
    }

    pub fn capture_window(_hwnd: WindowHandle) -> Option<FrameData> {
        None
    }

    pub fn capture_screenshot() -> Option<FrameData> {
        None
    }

    pub fn get_monitors() -> Vec<MonitorInfo> {
        vec![MonitorInfo {
            bounds: ScreenRect::new(0, 0, 1920, 1080),
            work_area: ScreenRect::new(0, 0, 1920, 1000),
            is_primary: true,
        }]
    }

    pub fn list_windows(
        _include: &[String],
        _exclude: &[String],
        _always_skip: &[(String, String)],
    ) -> Vec<(WindowHandle, String, String, ScreenRect)> {
        vec![]
    }

    pub fn check_duplicate_instances() -> bool {
        false
    }
}
