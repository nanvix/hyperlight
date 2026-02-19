/*
Copyright 2025  The Hyperlight Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

//! Time provider for FAT filesystem operations.

/// A time provider that always returns the FAT epoch (1980-01-01 00:00:00).
///
/// Guest VMs have no reliable clock, so we use a fixed timestamp.
/// The FAT epoch is the earliest representable FAT timestamp and clearly
/// indicates "timestamp not meaningful."
///
/// # Spec Reference
///
/// See spec §5.5.4 for rationale.
#[derive(Debug, Clone, Copy, Default)]
pub struct HyperlightTimeProvider;

impl fatfs::TimeProvider for HyperlightTimeProvider {
    fn get_current_date(&self) -> fatfs::Date {
        // 1980-01-01: FAT epoch
        fatfs::Date::new(1980, 1, 1)
    }

    fn get_current_date_time(&self) -> fatfs::DateTime {
        // 1980-01-01 00:00:00.000: FAT epoch
        fatfs::DateTime::new(fatfs::Date::new(1980, 1, 1), fatfs::Time::new(0, 0, 0, 0))
    }
}

#[cfg(test)]
mod tests {
    use fatfs::TimeProvider;

    use super::*;

    #[test]
    fn test_time_provider_returns_fat_epoch_date() {
        let provider = HyperlightTimeProvider;
        let date = provider.get_current_date();

        // FAT epoch is 1980-01-01
        assert_eq!(date.year, 1980);
        assert_eq!(date.month, 1);
        assert_eq!(date.day, 1);
    }

    #[test]
    fn test_time_provider_returns_fat_epoch_datetime() {
        let provider = HyperlightTimeProvider;
        let datetime = provider.get_current_date_time();

        // FAT epoch is 1980-01-01 00:00:00
        assert_eq!(datetime.date.year, 1980);
        assert_eq!(datetime.date.month, 1);
        assert_eq!(datetime.date.day, 1);
        assert_eq!(datetime.time.hour, 0);
        assert_eq!(datetime.time.min, 0);
        assert_eq!(datetime.time.sec, 0);
        assert_eq!(datetime.time.millis, 0);
    }

    #[test]
    fn test_time_provider_is_default() {
        // Verify Default derive works
        let _provider: HyperlightTimeProvider = Default::default();
    }

    #[test]
    fn test_time_provider_is_copy_clone() {
        let provider = HyperlightTimeProvider;
        let _copied = provider; // Copy
        let _cloned = provider.clone(); // Clone
    }
}
