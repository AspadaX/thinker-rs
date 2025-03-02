use anyhow::{Error, Result};

use console::Term;
use thinker_rs::{graph::Graph, inference::{Inference, InferenceParameters}, prompt::Instruction};

fn main() -> Result<(), Error> {
    let instructions: Vec<Instruction> = vec![
        Instruction::from_file("/Users/xinyubao/Documents/thinker-rs/instructions/thought.json")?,
        Instruction::from_file("/Users/xinyubao/Documents/thinker-rs/instructions/closeness_to_answer.json")?
    ];
    let terminal = Term::stdout();
    terminal.write_line(&console::style("Instructions loaded.").green().to_string())?;

    let parameters = InferenceParameters::new(
        3, 
        5.0,
        Some(3)
    );
    terminal.write_line(
        &console::style(
            format!(
                    "Inference will have max depth of {}", 
                    parameters.get_max_depth().unwrap_or(0)
                )
        )
                .green()
                .to_string()
    )?;

    let mut inference = Inference::new(
        "http://192.168.0.101:11434/v1".to_string(), 
        "1".to_string(), 
        instructions, 
        "How many vertical asymptotes does the graph of $y=\\frac{2}{x^2+x-6}$ have?".to_string(), 
        "mistral".to_string()
    );
    terminal.write_line(&console::style("Inference object created.").green().to_string())?;

    let graph: Graph = inference.run(parameters)?;
    terminal.write_line(&console::style(format!("Inference run completed. Graph: {:?}", graph)).green().to_string())?;
    
    graph.save("./graph.json")?;

    Ok(())
}
