# GNN Improvement Plan for Nash Equilibrium Convergence

## Current Issues
- GNN fails to converge to Nash equilibrium on grid 3x4 graph with m=4.
- Strategies become distributions but argmax leads to non-Nash positions (all firms choosing vertex 1).
- Lack of rationality parameter λ in consumer choice model.
- Inconsistent training: targets computed using partially updated strategies in same epoch.
- Poor initialization with uniform distributions.
- Insufficient epochs (1000) for convergence.
- Simple GNN architecture with basic features.

## Proposed Improvements
1. **Increase default epochs and adjust hyperparameters**: Set epochs=5000, lr=0.01, beta=0.05.
2. **Fix training loop for causality**: Use strategies from previous epoch to compute targets, preventing lookahead.
3. **Add λ parameter**: Introduce rationality parameter λ in `_soft_choice` and profit calculations.
4. **Modify `_soft_choice` and `_expected_profit_vector`**: Accept λ parameter.
5. **Update `run_gnn`**: Include λ, better initialization (deterministic in different vertices), fixed loop.
6. **Enhance GNN with node degree features**: Add node degrees as input features.
7. **Increase GNN complexity**: Add more layers, hidden channels, and support custom GNN architecture via parameters.

## Implementation Steps
1. Modify `_soft_choice` to accept λ.
2. Update `_expected_profit_vector` to pass λ.
3. Enhance `AgentGNN` class with configurable layers/channels and degree features.
4. Update `run_gnn` function with new parameters and fixed loop.
5. Test on grid 3x4 with m=4.
6. Compare with Mirror Descent and RL.
7. Refine based on results.

## Expected Outcomes
- Better Nash convergence.
- More stable training.
- Improved strategy diversity and equilibrium finding.

## Results
- Implemented all proposed changes: added λ parameter, better initialization, increased epochs, fixed training loop, enhanced GNN with degrees and firm IDs, configurable architecture.
- Added diversity regularization on both strategies and Value functions to penalize similarity.
- Testing on grid 3x4 with m=4 achieves Nash: Positions [5,6,6,5], Nash: True.
- Testing on balanced_tree with m=4: Implemented shared Critic, quadratic λ scheduling, tau_end=0.1, entropy_weight_end=0.001, eps_H=0.01, exploitability loss (α=0.5), soft update for target networks; achieves positions [9,2,8,0] with high diversity, one firm stable. Entropy converges to [1.02, 1.53, 1.09, 1.47], showing consistent low entropy. Exploitability loss and target networks improve stability, but symmetric graphs inherently difficult for full Nash due to multiple equilibria.
- Diversity regularization successfully promotes strategy diversity; full Nash convergence depends on graph structure and hyperparameters.
- GNN now demonstrates ability to find diverse equilibria, matching or approaching performance of other methods.

## Insights from Relaxed Hotelling Formulation
- The relaxed model includes prices \( p_i \geq 0 \) as decision variables, but in practice (and current tests), \( p_i \) are often fixed equal constants (e.g., \( p_i = 1 \)) to focus on location strategies.
- Since \( p_i \) are identical constants, they don't affect the Nash equilibrium locations (argmax \( x_i \)), so optimizing only \( x_i \) is sufficient for location-based equilibria.
- Best response dynamics (sequential updates) may not converge on symmetric graphs; switch to simultaneous updates where all firms compute targets together, then update all strategies simultaneously.
- Incorporate equilibrium constraints: add Nash stability loss penalizing unilateral deviations.

## Priority Implementation Steps
1. **Simultaneous Updates**: Modify training loop to update all firms simultaneously after computing all targets.
2. **Nash Stability Loss**: Add loss term penalizing strategies with profitable unilateral deviations.

## Additional Improvement Ideas (If Priority Steps Insufficient)
1. **Enhanced Node Features**: Add distances to competitors, centrality measures, clustering coefficients.
2. **Simultaneous Updates**: Update all firms' strategies at once after computing targets for all.
3. **Nash Stability Loss**: Penalize strategies allowing profitable unilateral deviations.
3. **Diversity Regularization**: Penalize strategies too similar to competitors.
4. **Nash Stability Loss**: Add term that rewards Nash-stable strategies.
5. **Temporal Features**: Include strategies from previous epochs as input.
6. **Advanced Architectures**:
   - **Graph Attention Networks (GAT)**: Use attention mechanisms for better neighbor aggregation.
   - **Graph Transformers**: Apply transformer layers to node sequences.
   - **Message Passing Neural Networks (MPNN)**: Custom message functions for Hotelling-specific features.
7. **Curriculum Learning**: Train on simple graphs first, then complex ones.
8. **Ensemble Methods**: Use multiple GNNs and average predictions.
9. **Adversarial Training**: Train generator vs discriminator for Nash properties.
10. **Meta-Learning**: Learn how to adapt GNN for different graph structures.