#[derive(Clone, Debug)]
pub struct GlobalPnrProgress {
    enabled: bool,
    module_name: String,
}

impl GlobalPnrProgress {
    pub fn new(enabled: bool, module_name: impl Into<String>) -> Self {
        Self {
            enabled,
            module_name: module_name.into(),
        }
    }

    pub fn stage(&self, step: usize, total: usize, name: impl AsRef<str>) {
        if self.enabled {
            eprintln!(
                "global pnr {}: [{step}/{total}] {}",
                self.module_name,
                name.as_ref()
            );
        }
    }

    pub fn item(&self, step: usize, total: usize, name: impl AsRef<str>) {
        if self.enabled {
            eprintln!("  [{step}/{total}] {}", name.as_ref());
        }
    }

    pub fn detail(&self, message: impl AsRef<str>) {
        if self.enabled {
            eprintln!("  {}", message.as_ref());
        }
    }
}
