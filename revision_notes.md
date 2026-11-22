# Revision notes

## Files and scope

- Preserved the original research_paper_phase2.docx unchanged and created research_paper_phase2_revised.docx.
- Created a matching PDF from the revised Word document using Microsoft Word.
- Created this note file.
- No Markdown source for the original paper was present in the project, so no revised Markdown copy was created.
- No project source code, checkpoints, training metrics, or experiment outputs were modified.

## Factual and terminology corrections

- Updated the title block with Vishva Pujara and Jaimin Sitapara, their department and university, and clickable email addresses. Revised the subtitle, abstract, introduction, and environment description to identify Phase 2 as completed work and its simplified rules as intentional.
- Reviewed every Phase 2 mention. They describe the implemented environment, completed experiments, or historical comparisons; no section presents Phase 2 as planned or awaiting implementation. No technical section required removal.
- Corrected initialization: each configured deck is scanned in its existing order and the first Pokémon is removed for the Active position before the remaining cards are shuffled. The configured order begins with Pikachu, so both players start with Pikachu Active; the agent does not choose it. Red receives four opening cards and has 15 cards remaining; Blue receives three and has 16 remaining.
- Recast the model input as a 50-dimensional observation vector and described the separate 70-element legal-action mask. The paper no longer calls the vector a complete canonical information-set representation.
- Clarified that the observation builder exposes the observing player’s visible hand and public board while exposing only the opponent’s hand and deck sizes.
- Rechecked the action mapping, game mechanics, model dimensions, Deep CFR self-play, depth-four explicit traversal, terminal continuation, and iteration-weighted masked cross-entropy against the current implementation.

## Evaluation metrics

- Filled the recorded average-reward and average-length values supplied with the experiment summaries for PPO, LSTM-PPO, Deep CFR, and the DQN random baseline.
- Kept the DQN best-checkpoint average reward as NR because no retained value was available.
- Preserved the three separate Deep CFR outcomes: greedy StrategyNetwork 13.0%, sampled StrategyNetwork 9.7%, and RegretNetwork with regret matching 53.6%.
- Training metric CSVs and performance plots are present, but machine-readable per-game evaluation logs are not. Aggregate evaluation metrics are reported from the recorded run summaries; some win rates are also visible in the plots. No new evaluation was run.

## Figures and references

- Added Figure 1, an implementation-based architecture and training/evaluation flow diagram.
- Added Figure 2, a win-rate chart with every game count shown; DQN’s 100-game protocol is distinguished from the other 1,500-game evaluations.
- Reformatted the seven existing sources in consistent IEEE numbered style and verified metadata against the publisher and proceedings records.
- Confirmed that Related Work cites sources [1]–[7] in the text and formatted the bibliography as a consistent numbered IEEE-style list.
- Added the official Kaggle card-data page as reference [8]. Its metadata identifies the competition host and title; access and use remain governed by competition rules. The restricted simulator source is not cited as an open-source dependency.

## Remaining items

- Preserve per-game evaluation records in future runs so aggregate metrics can be independently recalculated.
- Confirm that use and citation of competition card data comply with current Kaggle competition rules.

