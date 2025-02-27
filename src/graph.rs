use std::collections::HashSet;

use log::debug;
use serde::{Deserialize, Serialize};
use anyhow::{Error, Ok, Result};

use crate::{node::Node, prompt::Thought};

#[derive(Debug, Serialize, Deserialize)]
pub struct Graph {
    nodes: Vec<Node>
}

impl Graph {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn from_vec(nodes: Vec<Node>) -> Self {
        Self { nodes }
    }

    /// Create a new node in the graph
    pub fn new_node(
        &mut self,
        thought: Thought, 
    ) -> usize {
        let node = Node::new(
            thought.to_string()
        );
        self.nodes.push(node);

        // Return the latest node's index
        self.nodes.len() - 1
    }

    /// Child node's previous node is the parent node, and vice versa
    pub fn link_nodes(&mut self, parent_node: usize, child_node: usize) {
        if let Some(node) = self.nodes.get_mut(parent_node) {
            node.assign_next_node(child_node);
        }

        if let Some(node) = self.nodes.get_mut(child_node) {
            node.assign_previous_node(parent_node);
        }
    }

    /// Check if this branch is pruned
    pub fn is_pruned(&self, branch_indexes: Vec<usize>) -> bool {
        for (index, node) in self.nodes.iter().enumerate() {
            if branch_indexes.contains(&index) {
                if node.is_pruned() {
                    return true;
                }
            }
        }

        false
    }

    /// Get the only branch that is not pruned, if there
    /// are more than one that is not pruned, it will 
    /// return a None. 
    /// Note that the returned Graph object is a clone from 
    /// the original branch in the original graph, not
    /// a reference. 
    pub fn get_only_one_not_pruned(&self) -> Result<Option<Graph>, Error> {
        let branches: Vec<Vec<usize>> = self.get_branches();

        if branches.len() == 1 {
            let mut nodes = Vec::new();
            for (index, node) in self.nodes
                .iter()
                .enumerate() 
            {
                if branches[0].contains(&index) {
                    nodes.push(node.to_owned());
                }
            }

            return Ok(
                Some(
                    Graph::from_vec(nodes)
                )
            );
        }

        Ok(None)
    }

    /// Get all nodes on the same branch
    pub fn traverse(&self, start: usize) -> Vec<usize> {
        let mut visited: HashSet<usize> = HashSet::new();
        let mut stack: Vec<usize> = vec![start];
        let mut branch_nodes_indexes: Vec<usize> = Vec::new();

        while let Some(node_index) = stack.pop() {
            if visited.insert(node_index) {
                branch_nodes_indexes.push(node_index);
                if let Some(node) = self.nodes.get(node_index) {
                    if let Some(next_index) = node.get_next() {
                        stack.push(next_index);
                    }
                }
            }
        }

        branch_nodes_indexes
    }
    
    /// Access a node by its index
    pub fn access_node(&mut self, index: usize) -> Option<&mut Node> {
        self.nodes.get_mut(index)
    }
    
    /// Prune a branch
    pub fn prune_branch(&mut self, end_node_index: usize) {
        if let Some(node) = self.nodes.get_mut(end_node_index) {
            node.set_pruned();
        }
    }
    
    /// Get the depth of the graph. 
    /// The depth of a graph is the maximum depth of any branch in the graph.
    pub fn depth(&self) -> usize {
        let branches = self.get_branches();

        let mut maximum_depth: usize = 0;
        for branch in branches {
            let depth = branch.len();
            if depth > maximum_depth {
                maximum_depth = depth;
            }
        }
        
        maximum_depth
    }
    
    /// Get all branches in the graph.
    /// A branch is defined as a sequence of nodes starting from a node with no previous node
    /// and following the next nodes until the end of the branch.
    ///
    /// # Returns
    /// A vector of vectors, where each inner vector contains the indexes of nodes in a branch.
    pub fn get_branches(&self) -> Vec<Vec<usize>> {
        let mut start_nodes_index: Vec<usize> = Vec::new();
        let _ = self.nodes.iter().enumerate()
            .map(|(index, node)| {
                if node.get_previous().is_none() {
                    start_nodes_index.push(index);
                }
            });

        let mut branches: Vec<Vec<usize>> = Vec::new();
        for index in start_nodes_index {
            let branch = self.traverse(index);
            branches.push(branch);
        }

        branches
    }
    
    /// Get the end nodes indexes that are not yet pruned
    pub fn get_active_nodes(&self) -> Vec<usize> {
        let mut active_nodes: Vec<usize> = Vec::new();
        let _ = self.nodes.iter().enumerate()
            .map(|(index, node)| {
                if node.get_next().is_none() && !node.is_pruned() {
                    debug!("Active node found at index: {}", index);
                    active_nodes.push(index);
                }
            });

        debug!("Total active nodes: {}", active_nodes.len());
        active_nodes
    }

    /// Save the graph to a local position
    pub fn save(&self, path: &str) -> Result<(), Error> {
        let json = serde_json::to_string_pretty(&self)?;
        let _ = std::fs::write(path, json);

        Ok(())
    }

    /// Load a graph from a local position
    pub fn load(path: &str) -> Result<Self, Error> {
        let json: String = std::fs::read_to_string(path)?;

        Ok(serde_json::from_str::<Graph>(&json)?)
    }
}