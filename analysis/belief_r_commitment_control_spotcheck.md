# Belief-R Commitment-Control Spot Check

This memo records a deterministic 50-example inspection slice for the
first Belief-R commitment-control conversion.

## High-Level Notes

- `incremental_no_overturn` rows consistently map to `preserve`.
- `incremental_overturn_reasoning` rows consistently map to `replace` in
  the binary v1 label space.
- `full_info` rows share the same control label as their paired
  incremental example, so the dataset can supervise both control and
  full-information answer prediction without losing pair alignment.
- No `weaken` examples are introduced in v1; the label space remains
  3-way-ready but the current Belief-R proof-of-concept is binary on
  control decisions.

## Sampled Examples

### Example 1

- Example ID: `belief_r::790-strong::ponens::full_info`
- Split: `train`
- Condition: `full_info`
- Control label: `replace`
- Early commitment: `a :: Jessica hands John a pair of sunglasses.`
- Final answer: `c :: Jessica may or may not hand John a pair of sunglasses.`
- Pair metadata: `ponens / If-Event-Then-Event`

Early context:
```text
Premise 1: If John raises their hand to shield their eyes from the sun, then Jessica hands John a pair of sunglasses
Premise 2: John raises their hand to shield their eyes from the sun
```
Late evidence:
```text
Late Evidence 1: If John squints against the bright daylight then Jessica hands John a pair of sunglasses
```

### Example 2

- Example ID: `belief_r::728-strong::ponens::full_info`
- Split: `train`
- Condition: `full_info`
- Control label: `replace`
- Early commitment: `a :: John feels thrilled.`
- Final answer: `c :: John may or may not feel thrilled.`
- Pair metadata: `ponens / If-Event-Then-MentalState`

Early context:
```text
Premise 1: If John asks questions about a new topic, then John feels thrilled
Premise 2: John asks questions about a new topic
```
Late evidence:
```text
Late Evidence 1: If Jessica reveals her recent job promotion to John then John feels thrilled
```

### Example 3

- Example ID: `belief_r::547-strong::tollens::incremental`
- Split: `train`
- Condition: `incremental_overturn_reasoning`
- Control label: `replace`
- Early commitment: `b :: If John does not say the product saves time`
- Final answer: `c :: If John may or may not say the product saves time`
- Pair metadata: `tollens / If-Event-Then-MentalState`

Early context:
```text
Premise 1: If "If John says the product saves time, then Jessica will be glad to hear it's time-saving."
Premise 2: Jessica will not be glad to hear it's time-saving
```
Late evidence:
```text
Late Evidence 1: If Jessica values efficiency in her daily routine, then Jessica will be glad to hear it's time-saving
```

### Example 4

- Example ID: `belief_r::866-strong::ponens::full_info`
- Split: `train`
- Condition: `full_info`
- Control label: `replace`
- Early commitment: `a :: John speaks with Jessica.`
- Final answer: `c :: John may or may not speak with Jessica.`
- Pair metadata: `ponens / If-Event-Then-Event`

Early context:
```text
Premise 1: If John reads a book, then John speaks with Jessica
Premise 2: John reads a book
```
Late evidence:
```text
Late Evidence 1: If John discovers a typo in the novel he is reading then John speaks with Jessica
```

### Example 5

- Example ID: `belief_r::197-strong::ponens::incremental`
- Split: `train`
- Condition: `incremental_no_overturn`
- Control label: `preserve`
- Early commitment: `a :: Jessica sees it as sneaky.`
- Final answer: `a :: Jessica sees it as sneaky.`
- Pair metadata: `ponens / If-Event-Then-MentalState`

Early context:
```text
Premise 1: If John decides to sell their shared car without asking Jessica, then Jessica sees it as sneaky
Premise 2: John decides to sell their shared car without asking Jessica
```
Late evidence:
```text
Late Evidence 1: If John sells a jointly owned asset without consulting Jessica, then Jessica sees it as sneaky
```

### Example 6

- Example ID: `belief_r::833-strong::tollens::incremental`
- Split: `train`
- Condition: `incremental_overturn_reasoning`
- Control label: `replace`
- Early commitment: `b :: John does not create segments on a timeline to track daily tasks`
- Final answer: `c :: John may or may not create segments on a timeline to track daily tasks`
- Pair metadata: `tollens / If-Event-Then-Event`

Early context:
```text
Premise 1: If John creates segments on a timeline to track daily tasks, then Jessica can review the accuracy of these segments
Premise 2: Jessica cannot review the accuracy of these segments
```
Late evidence:
```text
Late Evidence 1: If Jessica has access to the same timeline data and has understanding of John's schedule then Jessica can review the accuracy of these segments
```

### Example 7

- Example ID: `belief_r::874-strong::ponens::full_info`
- Split: `train`
- Condition: `full_info`
- Control label: `replace`
- Early commitment: `a :: John feels joy.`
- Final answer: `c :: John may or may not feel joy.`
- Pair metadata: `ponens / If-Event-Then-MentalState`

Early context:
```text
Premise 1: If John decreases stress enough to solve fewer problems, then John feels joy
Premise 2: John decreases stress enough to solve fewer problems
```
Late evidence:
```text
Late Evidence 1: If John finishes his work early enough to spend the evening with Jessica then John feels joy
```

### Example 8

- Example ID: `belief_r::277-strong::tollens::incremental`
- Split: `train`
- Condition: `incremental_overturn_reasoning`
- Control label: `replace`
- Early commitment: `b :: John does not schedule a meeting`
- Final answer: `c :: John may or may not schedule a meeting`
- Pair metadata: `tollens / If-Event-Then-Event`

Early context:
```text
Premise 1: If John schedules a meeting, then John ensures Jessica can lead the discussion
Premise 2: John does not ensure Jessica can lead the discussion
```
Late evidence:
```text
Late Evidence 1: If John allocates the role of main speaker to Jessica for the agenda, then John ensures Jessica can lead the discussion
```

### Example 9

- Example ID: `belief_r::59-strong::ponens::incremental`
- Split: `train`
- Condition: `incremental_overturn_reasoning`
- Control label: `replace`
- Early commitment: `a :: John will gain friends.`
- Final answer: `c :: John may or may not gain friends.`
- Pair metadata: `ponens / If-Event-Then-Event`

Early context:
```text
Premise 1: If John teaches men effective communication, then John will gain friends
Premise 2: John teaches men effective communication
```
Late evidence:
```text
Late Evidence 1: If John consistently demonstrates empathy and genuine interest in Jessica's thoughts and feelings, then John will gain friends
```

### Example 10

- Example ID: `belief_r::267-strong::ponens::full_info`
- Split: `train`
- Condition: `full_info`
- Control label: `preserve`
- Early commitment: `a :: John will improve their craftsmanship.`
- Final answer: `a :: John will improve their craftsmanship.`
- Pair metadata: `ponens / If-Event-Then-Event`

Early context:
```text
Premise 1: If John attends a workshop, then John will improve their craftsmanship
Premise 2: John attends a workshop
```
Late evidence:
```text
Late Evidence 1: If John consistently practices the techniques demonstrated by master artisan Jessica, then John will improve their craftsmanship
```

### Example 11

- Example ID: `belief_r::259-strong::tollens::full_info`
- Split: `train`
- Condition: `full_info`
- Control label: `preserve`
- Early commitment: `b :: John does not want to be punctual`
- Final answer: `b :: John does not want to be punctual`
- Pair metadata: `tollens / If-Event-Then-Event`

Early context:
```text
Premise 1: If John wants to be punctual, then John sets an alarm to wake up early
Premise 2: John does not set an alarm to wake up early
```
Late evidence:
```text
Late Evidence 1: If John has an important meeting at 8 AM with Jessica who never tolerates lateness, then John sets an alarm to wake up early
```

### Example 12

- Example ID: `belief_r::275-strong::ponens::full_info`
- Split: `train`
- Condition: `full_info`
- Control label: `preserve`
- Early commitment: `a :: John contributes to constructing the birdhouse.`
- Final answer: `a :: John contributes to constructing the birdhouse.`
- Pair metadata: `ponens / If-Event-Then-Event`

Early context:
```text
Premise 1: If John designs a birdhouse, then John contributes to constructing the birdhouse
Premise 2: John designs a birdhouse
```
Late evidence:
```text
Late Evidence 1: If John selects the wood and Jessica measures and cuts it, then John contributes to constructing the birdhouse
```

### Example 13

- Example ID: `belief_r::484-strong::tollens::full_info`
- Split: `train`
- Condition: `full_info`
- Control label: `replace`
- Early commitment: `b :: John does not craft clay pots `
- Final answer: `c :: John may or may not craft clay pots`
- Pair metadata: `tollens / If-Event-Then-MentalState`

Early context:
```text
Premise 1: If John crafts clay pots, then they seek community, thus becoming a skilled potter
Premise 2: They do not seek community, thus not becoming a skilled potter
```
Late evidence:
```text
Late Evidence 1: If John diligently attends Jessica's renowned pottery workshops every weekend, then they seek community, thus becoming a skilled potter
```

### Example 14

- Example ID: `belief_r::589-strong::ponens::incremental`
- Split: `train`
- Condition: `incremental_no_overturn`
- Control label: `preserve`
- Early commitment: `a :: John includes a frame as a bonus.`
- Final answer: `a :: John includes a frame as a bonus.`
- Pair metadata: `ponens / If-Event-Then-Event`

Early context:
```text
Premise 1: If John sells Jessica's painting for $500 to establish its value, then John includes a frame as a bonus
Premise 2: John sells Jessica's painting for $500 to establish its value
```
Late evidence:
```text
Late Evidence 1: If John wishes to enhance the appeal of Jessica's artwork during the transaction then John includes a frame as a bonus
```

### Example 15

- Example ID: `belief_r::436-strong::tollens::full_info`
- Split: `train`
- Condition: `full_info`
- Control label: `replace`
- Early commitment: `b :: John does not master the sharpening process`
- Final answer: `c :: John may or may not master the sharpening process`
- Pair metadata: `tollens / If-Event-Then-Event`

Early context:
```text
Premise 1: If John masters the sharpening process, then John adjusts the grinder accurately and informs Jessica
Premise 2: John does not adjust the grinder accurately and does not inform Jessica
```
Late evidence:
```text
Late Evidence 1: If John has precisely measured the blade's angles and set the calibration tools correctly, then John adjusts the grinder accurately and informs Jessica
```

### Example 16

- Example ID: `belief_r::471-strong::ponens::full_info`
- Split: `train`
- Condition: `full_info`
- Control label: `replace`
- Early commitment: `a :: Jessica will remain with the book club members.`
- Final answer: `c :: Jessica may or may not remain with the book club members.`
- Pair metadata: `ponens / If-Event-Then-Event`

Early context:
```text
Premise 1: If John attends the local book club meeting, then Jessica will remain with the book club members
Premise 2: John attends the local book club meeting
```
Late evidence:
```text
Late Evidence 1: If Jessica enjoys the lively literary discussions at the weekly gatherings, then Jessica will remain with the book club members
```

### Example 17

- Example ID: `belief_r::389-strong::tollens::full_info`
- Split: `train`
- Condition: `full_info`
- Control label: `replace`
- Early commitment: `b :: John does not understand each detail`
- Final answer: `c :: John may or may not understand each detail`
- Pair metadata: `tollens / If-Event-Then-Event`

Early context:
```text
Premise 1: If John understands each detail, then Jessica will share extra details
Premise 2: Jessica will not share extra details
```
Late evidence:
```text
Late Evidence 1: If Jessica notices that her explanations are fully comprehended and met with insightful questions, then Jessica will share extra details
```

### Example 18

- Example ID: `belief_r::704-strong::ponens::full_info`
- Split: `train`
- Condition: `full_info`
- Control label: `replace`
- Early commitment: `a :: Jessica's job may be at risk.`
- Final answer: `c :: Jessica's job may or may not be at risk.`
- Pair metadata: `ponens / If-Event-Then-Event`

Early context:
```text
Premise 1: If John avoids studying the financial crisis in detail, then Jessica's job may be at risk
Premise 2: John avoids studying the financial crisis in detail
```
Late evidence:
```text
Late Evidence 1: If the chief financial analyst at a major bank neglects to analyze emerging economic downturns then Jessica's job may be at risk
```

### Example 19

- Example ID: `belief_r::13-strong::tollens::incremental`
- Split: `train`
- Condition: `incremental_overturn_reasoning`
- Control label: `replace`
- Early commitment: `b :: John does not lie to avoid punishment`
- Final answer: `c :: John may or may not lie to avoid punishment`
- Pair metadata: `tollens / If-Event-Then-MentalState`

Early context:
```text
Premise 1: If John lies to avoid punishment, then Jessica gets mad
Premise 2: Jessica does not get mad
```
Late evidence:
```text
Late Evidence 1: If he breaks her trust by deceiving her, then Jessica gets mad
```

### Example 20

- Example ID: `belief_r::23-strong::tollens::incremental`
- Split: `train`
- Condition: `incremental_overturn_reasoning`
- Control label: `replace`
- Early commitment: `b :: If John does not leverage their clout to get perks from friends `
- Final answer: `c :: If John may or may not leverage their clout to get perks from friends`
- Pair metadata: `tollens / If-Event-Then-Event`

Early context:
```text
Premise 1: If "If John leverages their clout to get perks from friends, then John will request additional favors."
Premise 2: John will not request additional favors
```
Late evidence:
```text
Late Evidence 1: If John frequently asks Jessica for help without offering anything in return, then John will request additional favors
```

### Example 21

- Example ID: `belief_r::390-strong::ponens::incremental`
- Split: `dev`
- Condition: `incremental_overturn_reasoning`
- Control label: `replace`
- Early commitment: `a :: Jessica inquires for clarity.`
- Final answer: `c :: Jessica may or may not inquire for clarity.`
- Pair metadata: `ponens / If-Event-Then-Event`

Early context:
```text
Premise 1: If John understands the topic fully, then Jessica inquires for clarity
Premise 2: John understands the topic fully
```
Late evidence:
```text
Late Evidence 1: If John provides a detailed explanation with complex terminology, then Jessica inquires for clarity
```

### Example 22

- Example ID: `belief_r::333-strong::ponens::full_info`
- Split: `dev`
- Condition: `full_info`
- Control label: `replace`
- Early commitment: `a :: John understands fruit varieties better`
- Final answer: `c :: John may or may not understand fruit varieties better`
- Pair metadata: `ponens / If-Event-Then-Event`

Early context:
```text
Premise 1: If John tells Jessica the difference between apples and oranges, then John understands fruit varieties better
Premise 2: John tells Jessica the difference between apples and oranges
```
Late evidence:
```text
Late Evidence 1: If John can accurately categorize six types of apples and four kinds of oranges during a discussion, then John understands fruit varieties better
```

### Example 23

- Example ID: `belief_r::811-strong::tollens::incremental`
- Split: `dev`
- Condition: `incremental_overturn_reasoning`
- Control label: `replace`
- Early commitment: `b :: John does not email a bill draft to the Legislation Committee`
- Final answer: `c :: John may or may not email a bill draft to the Legislation Committee`
- Pair metadata: `tollens / If-Event-Then-Event`

Early context:
```text
Premise 1: If John emails a bill draft to the Legislation Committee, then John will write a summary document
Premise 2: John will not write a summary document
```
Late evidence:
```text
Late Evidence 1: If John receives feedback from Jessica on the proposed legal text then John will write a summary document
```

### Example 24

- Example ID: `belief_r::581-strong::tollens::incremental`
- Split: `dev`
- Condition: `incremental_overturn_reasoning`
- Control label: `replace`
- Early commitment: `b :: John does not testify against their co-conspirators during their own trial`
- Final answer: `c :: John may or may not testify against their co-conspirators during their own trial`
- Pair metadata: `tollens / If-Event-Then-Event`

Early context:
```text
Premise 1: If John testifies against their co-conspirators during their own trial, then Jessica won't be investigated
Premise 2: Jessica will be investigated
```
Late evidence:
```text
Late Evidence 1: If full immunity for all involved parties is granted in exchange for the key evidence provided by John, then Jessica won't be investigated
```

### Example 25

- Example ID: `belief_r::385-strong::ponens::full_info`
- Split: `dev`
- Condition: `full_info`
- Control label: `preserve`
- Early commitment: `a :: Jessica poses queries.`
- Final answer: `a :: Jessica poses queries.`
- Pair metadata: `ponens / If-Event-Then-Event`

Early context:
```text
Premise 1: If John understands the topic fully, then Jessica poses queries
Premise 2: John understands the topic fully
```
Late evidence:
```text
Late Evidence 1: If John explains the subject without leaving any details unclear, then Jessica poses queries
```

### Example 26

- Example ID: `belief_r::895-strong::ponens::incremental`
- Split: `dev`
- Condition: `incremental_overturn_reasoning`
- Control label: `replace`
- Early commitment: `a :: John replaces it with a new bulb.`
- Final answer: `c :: John may or may not replace it with a new bulb.`
- Pair metadata: `ponens / If-Event-Then-Event`

Early context:
```text
Premise 1: If John has a burned-out light bulb, then John replaces it with a new bulb
Premise 2: John has a burned-out light bulb
```
Late evidence:
```text
Late Evidence 1: If John notices the room is darker than usual and Jessica confirms the lamp no longer works then John replaces it with a new bulb
```

### Example 27

- Example ID: `belief_r::318-strong::tollens::incremental`
- Split: `dev`
- Condition: `incremental_no_overturn`
- Control label: `preserve`
- Early commitment: `b :: John did not forget Anthony's bag at Jessica's house`
- Final answer: `b :: John did not forget Anthony's bag at Jessica's house`
- Pair metadata: `tollens / If-Event-Then-Event`

Early context:
```text
Premise 1: If John forgot Anthony's bag at Jessica's house, then Jessica had to take a shower
Premise 2: Jessica did not have to take a shower
```
Late evidence:
```text
Late Evidence 1: If Jessica was sprayed by a skunk while retrieving the bag John left, then Jessica had to take a shower
```

### Example 28

- Example ID: `belief_r::463-strong::tollens::full_info`
- Split: `dev`
- Condition: `full_info`
- Control label: `replace`
- Early commitment: `b :: John does not combine the two halves of a broken plate`
- Final answer: `c :: John may or may not combine the two halves of a broken plate`
- Pair metadata: `tollens / If-Event-Then-MentalState`

Early context:
```text
Premise 1: If John combines the two halves of a broken plate, then John feels content
Premise 2: John does not feel content
```
Late evidence:
```text
Late Evidence 1: If John successfully restores a cherished heirloom that Jessica believed was irreparable, then John feels content
```

### Example 29

- Example ID: `belief_r::939-strong::ponens::full_info`
- Split: `dev`
- Condition: `full_info`
- Control label: `replace`
- Early commitment: `a :: Jessica smiles at the choice.`
- Final answer: `c :: Jessica may or may not smile at the choice.`
- Pair metadata: `ponens / If-Event-Then-MentalState`

Early context:
```text
Premise 1: If John selects a precision screwdriver to fix a watch, then Jessica smiles at the choice
Premise 2: John selects a precision screwdriver to fix a watch
```
Late evidence:
```text
Late Evidence 1: If John chooses the ideal tool for delicate tasks then Jessica smiles at the choice
```

### Example 30

- Example ID: `belief_r::844-strong::tollens::full_info`
- Split: `dev`
- Condition: `full_info`
- Control label: `replace`
- Early commitment: `b :: John does not receive a marriage proposal from the prince`
- Final answer: `c :: John may or may not receive a marriage proposal from the prince`
- Pair metadata: `tollens / If-Event-Then-Event`

Early context:
```text
Premise 1: If John receives a marriage proposal from the prince, then Jessica will purchase a house
Premise 2: Jessica will not purchase a house
```
Late evidence:
```text
Late Evidence 1: If Jessica inherits a considerable sum from Cousin A's estate then Jessica will purchase a house
```

### Example 31

- Example ID: `belief_r::888-strong::ponens::full_info`
- Split: `dev`
- Condition: `full_info`
- Control label: `replace`
- Early commitment: `a :: John will start sleeping better.`
- Final answer: `c :: John may or may not start sleeping better.`
- Pair metadata: `ponens / If-Event-Then-Event`

Early context:
```text
Premise 1: If John cuts their daily screen time to 1 hour, then John will start sleeping better
Premise 2: John cuts their daily screen time to 1 hour
```
Late evidence:
```text
Late Evidence 1: If John replaces evening electronic device usage with reading a book then John will start sleeping better
```

### Example 32

- Example ID: `belief_r::59-strong::tollens::incremental`
- Split: `dev`
- Condition: `incremental_overturn_reasoning`
- Control label: `replace`
- Early commitment: `b :: John does not teach men effective communication`
- Final answer: `c :: John may or may not teach men effective communication`
- Pair metadata: `tollens / If-Event-Then-Event`

Early context:
```text
Premise 1: If John teaches men effective communication, then John will gain friends
Premise 2: John will not gain friends
```
Late evidence:
```text
Late Evidence 1: If John consistently demonstrates empathy and genuine interest in Jessica's thoughts and feelings, then John will gain friends
```

### Example 33

- Example ID: `belief_r::349-strong::ponens::incremental`
- Split: `dev`
- Condition: `incremental_overturn_reasoning`
- Control label: `replace`
- Early commitment: `a :: They intend to be fair.`
- Final answer: `c :: They may or may not intend to be fair.`
- Pair metadata: `ponens / If-Event-Then-Event`

Early context:
```text
Premise 1: If John persuades people for a job promotion, then they intend to be fair
Premise 2: John persuades people for a job promotion
```
Late evidence:
```text
Late Evidence 1: If John consistently advocates for Jessica's contributions to be recognized equally alongside his own, then they intend to be fair
```

### Example 34

- Example ID: `belief_r::484-strong::ponens::full_info`
- Split: `dev`
- Condition: `full_info`
- Control label: `replace`
- Early commitment: `a :: They seek community, thus becoming a skilled potter.`
- Final answer: `c :: They may or may not seek community, thus possibly becoming a skilled potter.`
- Pair metadata: `ponens / If-Event-Then-MentalState`

Early context:
```text
Premise 1: If John crafts clay pots, then they seek community, thus becoming a skilled potter
Premise 2: John crafts clay pots
```
Late evidence:
```text
Late Evidence 1: If John diligently attends Jessica's renowned pottery workshops every weekend, then they seek community, thus becoming a skilled potter
```

### Example 35

- Example ID: `belief_r::830-strong::ponens::incremental`
- Split: `dev`
- Condition: `incremental_overturn_reasoning`
- Control label: `replace`
- Early commitment: `a :: Jessica finds the lessons more comprehensible.`
- Final answer: `c :: Jessica may or may not find the lessons more comprehensible.`
- Pair metadata: `ponens / If-Event-Then-Event`

Early context:
```text
Premise 1: If John divides the textbook into chapters to get a better grasp of historical events, then Jessica finds the lessons more comprehensible
Premise 2: John divides the textbook into chapters to get a better grasp of historical events
```
Late evidence:
```text
Late Evidence 1: If John organizes the content into smaller, focused sections to simplify complex histories then Jessica finds the lessons more comprehensible
```

### Example 36

- Example ID: `belief_r::824-strong::tollens::incremental`
- Split: `test`
- Condition: `incremental_overturn_reasoning`
- Control label: `replace`
- Early commitment: `b :: John does not divide the text into sentences to count them.`
- Final answer: `c :: John may or may not divide the text into sentences to count them.`
- Pair metadata: `tollens / If-Event-Then-Event`

Early context:
```text
Premise 1: If John divides the text into sentences to count them, then John can organize the sentences by topic
Premise 2: John cannot organize the sentences by topic
```
Late evidence:
```text
Late Evidence 1: If John has systematically labeled each sentence with a unique identifier based on its content for Jessica's review then John can organize the sentences by topic
```

### Example 37

- Example ID: `belief_r::29-strong::ponens::incremental`
- Split: `test`
- Condition: `incremental_overturn_reasoning`
- Control label: `replace`
- Early commitment: `a :: John opens a coffee shop.`
- Final answer: `c :: John may or may not open a coffee shop.`
- Pair metadata: `ponens / If-Event-Then-Event`

Early context:
```text
Premise 1: If John uses their money to buy information for a secret plan, then John opens a coffee shop
Premise 2: John uses their money to buy information for a secret plan
```
Late evidence:
```text
Late Evidence 1: If John learns from Jessica that the most lucrative business in their area involves selling rare Ethiopian beans, then John opens a coffee shop
```

### Example 38

- Example ID: `belief_r::900-strong::ponens::incremental`
- Split: `test`
- Condition: `incremental_overturn_reasoning`
- Control label: `replace`
- Early commitment: `a :: John will better understand Jessica's strategy.`
- Final answer: `c :: John may or may not better understand Jessica's strategy.`
- Pair metadata: `ponens / If-Event-Then-Event`

Early context:
```text
Premise 1: If John mirrors Jessica's chess opening, then John will better understand Jessica's strategy
Premise 2: John mirrors Jessica's chess opening
```
Late evidence:
```text
Late Evidence 1: If John studies the same tactical patterns and positions that Jessica frequently employs in her games then John will better understand Jessica's strategy
```

### Example 39

- Example ID: `belief_r::917-strong::ponens::incremental`
- Split: `test`
- Condition: `incremental_overturn_reasoning`
- Control label: `replace`
- Early commitment: `a :: John will win the match.`
- Final answer: `c :: John may or may not win the match.`
- Pair metadata: `ponens / If-Event-Then-Event`

Early context:
```text
Premise 1: If John plays chess with Jessica by following the rules, then John will win the match
Premise 2: John plays chess with Jessica by following the rules
```
Late evidence:
```text
Late Evidence 1: If John consistently makes strategic moves while Jessica makes several critical errors then John will win the match
```

### Example 40

- Example ID: `belief_r::33-strong::ponens::incremental`
- Split: `test`
- Condition: `incremental_overturn_reasoning`
- Control label: `replace`
- Early commitment: `a :: Jessica receives compliments. Jessica feels flattered.`
- Final answer: `c :: Jessica may or may not receive compliments. Jessica may or may not feel flattered.`
- Pair metadata: `ponens / If-Event-Then-MentalState`

Early context:
```text
Premise 1: If John compliments men. John feels happy., then Jessica receives compliments. Jessica feels flattered
Premise 2: John compliments men. John feels happy
```
Late evidence:
```text
Late Evidence 1: If Jessica is praised by her peers for her achievements, then Jessica receives compliments and feels flattered
```

### Example 41

- Example ID: `belief_r::902-strong::tollens::full_info`
- Split: `test`
- Condition: `full_info`
- Control label: `replace`
- Early commitment: `b :: John does not follow the rules of chess as played by Jessica to understand the strategy`
- Final answer: `c :: John may or may not follow the rules of chess as played by Jessica to understand the strategy`
- Pair metadata: `tollens / If-Event-Then-Event`

Early context:
```text
Premise 1: If John follows the rules of chess as played by Jessica to understand the strategy, then Jessica obtains a strategic upper hand
Premise 2: Jessica does not obtain a strategic upper hand
```
Late evidence:
```text
Late Evidence 1: If John constantly reacts to Jessica's moves without anticipating future plays then Jessica obtains a strategic upper hand
```

### Example 42

- Example ID: `belief_r::9-strong::tollens::incremental`
- Split: `test`
- Condition: `incremental_overturn_reasoning`
- Control label: `replace`
- Early commitment: `b :: John does not escape a locked room by picking the lock`
- Final answer: `c :: John may or may not escape a locked room by picking the lock`
- Pair metadata: `tollens / If-Event-Then-MentalState`

Early context:
```text
Premise 1: If John escapes a locked room by picking the lock, then John smiles
Premise 2: John does not smile
```
Late evidence:
```text
Late Evidence 1: If John finally hears the satisfying click of the tumbler after Jessica teaches him the delicate art of lockpicking, then John smiles
```

### Example 43

- Example ID: `belief_r::861-strong::ponens::incremental`
- Split: `test`
- Condition: `incremental_overturn_reasoning`
- Control label: `replace`
- Early commitment: `a :: John listens carefully.`
- Final answer: `c :: John may or may not listen carefully.`
- Pair metadata: `ponens / If-Event-Then-MentalState`

Early context:
```text
Premise 1: If John focuses on listening, then John listens carefully
Premise 2: John focuses on listening
```
Late evidence:
```text
Late Evidence 1: If Jessica is sharing a detailed account of her day and John nods and responds appropriately to complex points she makes then John listens carefully
```

### Example 44

- Example ID: `belief_r::670-strong::ponens::incremental`
- Split: `test`
- Condition: `incremental_overturn_reasoning`
- Control label: `replace`
- Early commitment: `a :: Jessica will express gratitude to John.`
- Final answer: `c :: Jessica may or may not express gratitude to John.`
- Pair metadata: `ponens / If-Event-Then-Event`

Early context:
```text
Premise 1: If John discovers a leak in the laboratory, then Jessica will express gratitude to John
Premise 2: John discovers a leak in the laboratory
```
Late evidence:
```text
Late Evidence 1: If John alerts the safety team about a potential hazard in the workspace then Jessica will express gratitude to John
```

### Example 45

- Example ID: `belief_r::736-strong::ponens::incremental`
- Split: `test`
- Condition: `incremental_overturn_reasoning`
- Control label: `replace`
- Early commitment: `a :: Jessica, who loves hiking, will benefit from the new map John creates.`
- Final answer: `c :: Jessica, who loves hiking, may or may not benefit from the new map John creates.`
- Pair metadata: `ponens / If-Event-Then-MentalState`

Early context:
```text
Premise 1: If John explores every unmapped trail in their local park, then Jessica, who loves hiking, will benefit from the new map John creates
Premise 2: John explores every unmapped trail in their local park
```
Late evidence:
```text
Late Evidence 1: If John diligently charts each previously undocumented pathway in the surrounding wilderness area then Jessica, who loves hiking, will benefit from the new map John creates
```

### Example 46

- Example ID: `belief_r::132-strong::tollens::full_info`
- Split: `test`
- Condition: `full_info`
- Control label: `replace`
- Early commitment: `b :: John does not pay for a loaf of bread`
- Final answer: `c :: John may or may not pay for a loaf of bread`
- Pair metadata: `tollens / If-Event-Then-Event`

Early context:
```text
Premise 1: If John pays for a loaf of bread, then John can take the bread home
Premise 2: John cannot take the bread home
```
Late evidence:
```text
Late Evidence 1: If Jessica at the register hands John a receipt for his purchase, then John can take the bread home
```

### Example 47

- Example ID: `belief_r::350-strong::tollens::full_info`
- Split: `test`
- Condition: `full_info`
- Control label: `replace`
- Early commitment: `b :: John does not persuade the committee for a promotion`
- Final answer: `c :: John may or may not persuade the committee for a promotion`
- Pair metadata: `tollens / If-Event-Then-Event`

Early context:
```text
Premise 1: If John persuades the committee for a promotion, then Jessica will follow orders
Premise 2: Jessica will not follow orders
```
Late evidence:
```text
Late Evidence 1: If John becomes the new team leader, then Jessica will follow orders
```

### Example 48

- Example ID: `belief_r::923-strong::ponens::full_info`
- Split: `test`
- Condition: `full_info`
- Control label: `replace`
- Early commitment: `a :: They encounter Jessica, whom they intend to introduce to their sibling.`
- Final answer: `c :: They encounter Jessica, whom they may or may not intend to introduce to their sibling.`
- Pair metadata: `ponens / If-Event-Then-Event`

Early context:
```text
Premise 1: If John arrives at the local pub, then they encounter Jessica, whom they intend to introduce to their sibling
Premise 2: John arrives at the local pub
```
Late evidence:
```text
Late Evidence 1: If John is at his sister's favorite coffee shop for trivia night then they encounter Jessica, whom they intend to introduce to their sibling
```

### Example 49

- Example ID: `belief_r::639-strong::tollens::full_info`
- Split: `test`
- Condition: `full_info`
- Control label: `preserve`
- Early commitment: `b :: John does not optimize the use of stationery supplies in the office`
- Final answer: `b :: John does not optimize the use of stationery supplies in the office`
- Pair metadata: `tollens / If-Event-Then-Event`

Early context:
```text
Premise 1: If John optimizes the use of stationery supplies in the office, then Jessica thanks John for their cost-saving efforts
Premise 2: Jessica does not thank John for their cost-saving efforts
```
Late evidence:
```text
Late Evidence 1: If John implements a new inventory system that reduces the purchase frequency of office items then Jessica thanks John for their cost-saving efforts
```

### Example 50

- Example ID: `belief_r::512-strong::tollens::full_info`
- Split: `test`
- Condition: `full_info`
- Control label: `replace`
- Early commitment: `b :: John does not read the Bible`
- Final answer: `c :: John may or may not read the Bible`
- Pair metadata: `tollens / If-Event-Then-Event`

Early context:
```text
Premise 1: If John reads the Bible, then John lives by the Sermon on the Mount and attends Sunday service
Premise 2: John does not live by the Sermon on the Mount and does not attend Sunday service
```
Late evidence:
```text
Late Evidence 1: If John follows Jesus's teachings and joins weekly worship, then John lives by the Sermon on the Mount and attends Sunday service
```
