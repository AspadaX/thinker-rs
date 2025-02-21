use std::fmt::Display;

use anyhow::{Error, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub enum PromptType {
    Thought,
    ClosenessToAnswer,
    Reflection
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Instruction {
    goal: String,
    example: String,
    is_output_json: bool,
    prompt_type: PromptType,
}

impl Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_output_json {
            f.write_str(
                &format!(
                    "Output in json. \nGoal: \n{}\nExample: \n{}", 
                    self.goal, self.example
                )
            );

            return Ok(());
        }

        f.write_str(
            &format!("Goal: \n{}\nExample: \n{}", self.goal, self.example)
        );

        return Ok(());
    }
}

impl Instruction {
    pub fn from_file(path: &str) -> Result<Self, Error> {
        let file_string: String = std::fs::read_to_string(path)?;
        let instruction: Instruction = serde_json::from_str(&file_string)?;

        Ok(instruction)
    }

    pub fn access_prompt_type(&self) -> &PromptType {
        &self.prompt_type
    }
}