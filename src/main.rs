use anyhow::{Error, Result};

use thinker_rs::{graph::Graph, inference::{Inference, InferenceParameters}, prompt::Instruction};

fn main() -> Result<(), Error> {
    // Setup a logger
    simple_logger::SimpleLogger::new().env().init().unwrap();
    
    let instructions: Vec<Instruction> = vec![
        Instruction::from_file("/Users/xinyubao/Documents/thinker-rs/instructions/thought.json")?,
        Instruction::from_file("/Users/xinyubao/Documents/thinker-rs/instructions/closeness_to_answer.json")?
    ];
    log::info!("Instructions loaded: {:?}", instructions);

    let parameters = InferenceParameters::new(
        3, 
        5.0
    );
    log::info!("Inference parameters set: {:?}", parameters);

    let mut inference = Inference::new(
        "http://192.168.0.101:11434/v1".to_string(), 
        "1".to_string(), 
        instructions, 
        "How to get to New York from Nanjing with the cheapest cost?".to_string(), 
        "mistral".to_string()
    );
    log::info!("Inference object created: {:?}", inference);

    let graph: Graph = inference.run(parameters)?;
    log::info!("Inference run completed. Graph: {:?}", graph);

    Ok(())
}
