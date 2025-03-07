// Export modules
pub mod config;
pub mod image_processing;
pub mod categorize;
pub mod benchmark;
pub mod confusion_matrix;
pub mod natural_categorization;
pub mod natural_categorization_split;

// Re-export commonly used types
pub use config::{Config, RunMode, MatchingMethod};
