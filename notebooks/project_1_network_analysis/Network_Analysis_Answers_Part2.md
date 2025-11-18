Network Analysis Questions - Part 2
College Football Network Centrality Analysis

Student: Chuba Yevheniy
Course: Art of Data Visualization
Assignment: Network Visualization Analysis

================================================================================

Question 1a: What does degree centrality denote in this network?

In the college football network, degree centrality represents the number of games a team played against other teams in the dataset. Specifically, it measures how many direct connections (edges) each team (node) has in the network.

A team with high degree centrality has played games against many different opponents, making it highly connected within the college football landscape. The degree centrality values are normalized, so they represent the proportion of possible connections each team has made.

Key insight: Teams in the high degree centrality subgraph (12 nodes) are those that played the most games against different opponents during the season.

================================================================================

Question 1b: What does betweenness centrality denote in this network?

Betweenness centrality in this network measures how often a team appears on the shortest path between other pairs of teams. It identifies teams that serve as "bridges" or connectors in the college football network structure.

A team with high betweenness centrality acts as an important intermediary - many of the shortest paths between other teams pass through this team. This could indicate teams that:
• Connect different conferences or regions
• Serve as crucial links in the overall network structure
• Play a strategic role in the interconnectedness of college football

Key insight: Teams with high betweenness centrality (57 nodes in our subgraph) facilitate connections between different parts of the network, even if they don't necessarily play the most games.

================================================================================

Question 1c: Structural differences between the two network subgraphs

STRUCTURAL DIFFERENCES:

1. Size Difference:
   • Degree centrality subgraph: 12 nodes (highly selective)
   • Betweenness centrality subgraph: 57 nodes (much more inclusive)

2. Network Density:
   • The degree centrality subgraph appears more sparse and linear
   • The betweenness centrality subgraph is much denser with more interconnections

3. Connectivity Patterns:
   • Degree centrality subgraph shows simpler connection patterns
   • Betweenness centrality subgraph reveals complex, multi-layered relationships

IMPLICATIONS FOR DEGREE VS. BETWEENNESS CENTRALITY:

Degree Centrality Implications:
• Only a small elite group of teams (12 out of 115) played significantly more games than average
• High degree centrality is relatively rare in this network
• These teams likely represent major programs that schedule many non-conference games

Betweenness Centrality Implications:
• Many more teams (57 out of 115) serve important structural roles as bridges
• Betweenness centrality captures a broader range of network importance
• Teams don't need to play many games to be structurally important as connectors
• Geographic and conference positioning may influence betweenness more than raw game count

Overall: The dramatic size difference (12 vs 57 nodes) demonstrates that degree and betweenness centrality measure fundamentally different aspects of network importance.

================================================================================

Question 1d: Penn State vs. Ohio State Comparison

Based on the network subgraphs produced in Part 1:

PENN STATE:
• Appears in the degree centrality subgraph (left, blue visualization)
• This indicates Penn State played a high number of games against different opponents
• Shows Penn State as having many direct connections in the network

OHIO STATE:
• Appears in the betweenness centrality subgraph (right, coral visualization)
• This indicates Ohio State serves as an important bridge/connector in the network structure
• Ohio State facilitates connections between different parts of the college football network

KEY COMPARISON:
• Penn State demonstrates HIGH CONNECTIVITY (many direct relationships)
• Ohio State demonstrates HIGH STRUCTURAL IMPORTANCE (serves as a bridge between other teams)
• Penn State's strength is in the volume of connections, while Ohio State's strength is in its strategic positioning within the network structure

This difference suggests that Penn State may have scheduled more diverse opponents, while Ohio State occupies a more central position that connects different regions or conferences in college football.

================================================================================

CONCLUSION:

The network analysis reveals that degree and betweenness centrality capture different but complementary aspects of team importance in college football, with Penn State excelling in direct connectivity and Ohio State excelling in structural bridging roles.