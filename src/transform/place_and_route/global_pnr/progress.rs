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
        crate::snapshot::record(crate::snapshot::SnapshotEvent::Stage {
            module: self.module_name.clone(),
            step,
            total,
            name: name.as_ref().to_owned(),
        });
        if self.enabled {
            tracing::info!(
                target: "redstone_compiler::global_pnr",
                module = %self.module_name,
                step,
                total,
                stage = name.as_ref(),
                "global PnR stage"
            );
        }
    }

    pub fn item(&self, step: usize, total: usize, name: impl AsRef<str>) {
        if self.enabled {
            tracing::trace!(
                target: "redstone_compiler::global_pnr",
                module = %self.module_name,
                step,
                total,
                item = name.as_ref(),
                "global PnR item"
            );
        }
    }

    pub fn detail(&self, message: impl AsRef<str>) {
        if self.enabled {
            tracing::debug!(
                target: "redstone_compiler::global_pnr",
                module = %self.module_name,
                "{}",
                message.as_ref()
            );
        }
    }

    pub fn attempt(&self, step: usize, total: usize, name: impl AsRef<str>) {
        if self.enabled {
            tracing::debug!(
                target: "redstone_compiler::global_pnr",
                module = %self.module_name,
                attempt = step,
                total,
                strategy = name.as_ref(),
                "global PnR attempt"
            );
        }
    }

    pub fn summary(&self, message: impl AsRef<str>) {
        if self.enabled {
            tracing::info!(
                target: "redstone_compiler::global_pnr",
                module = %self.module_name,
                "{}",
                message.as_ref()
            );
        }
    }

    pub fn warning(&self, message: impl AsRef<str>) {
        if self.enabled {
            tracing::warn!(
                target: "redstone_compiler::global_pnr",
                module = %self.module_name,
                "{}",
                message.as_ref()
            );
        }
    }
}
