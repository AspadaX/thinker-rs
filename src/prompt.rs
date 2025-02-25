use std::{fmt::Display, str::FromStr};

use anyhow::{Error, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum PromptType {
    Thought,
    ClosenessToAnswer,
    Reflection
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Instruction {
    goal: String,
    example: String,
    json_format: Option<String>,
    prompt_type: PromptType,
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

impl FromStr for Instruction {
    type Err = Error;
    
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(serde_json::from_str::<Instruction>(&s)?)
    }
}

impl Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(json) = &self.json_format {
            f.write_str(
                &format!(
                    "Output in json. \nGoal: \n{}\nJSON Format: \n{}", 
                    self.goal, json
                )
            )?;

            return Ok(());
        }

        f.write_str(
            &format!("Goal: \n{}\nExample: \n{}", self.goal, self.example)
        )?;

        return Ok(());
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct Thought {
    reasoning: String,
    content: String
}

impl FromStr for Thought {
    type Err = Error;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(serde_json::from_str::<Thought>(s)?)
    }
}

impl Display for Thought {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.content)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClosenessToAnswer {
    score: f32
}

impl FromStr for ClosenessToAnswer {
    type Err = Error;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(serde_json::from_str::<ClosenessToAnswer>(s)?)
    }
}

impl Display for ClosenessToAnswer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.score.to_string())
    }
}

impl From<ClosenessToAnswer> for f32 {
    fn from(c: ClosenessToAnswer) -> Self {
        c.score
    }
}