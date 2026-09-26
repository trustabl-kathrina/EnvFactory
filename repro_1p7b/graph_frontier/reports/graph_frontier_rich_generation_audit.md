# Graph-Frontier-Rich generation audit

## Final verdict

`GRAPH_FRONTIER_RICH_POOL_READY`

## A. ToolGraph capacity

The graph contains 683 tools, 4633 parameters,
11067 NetworkX edges, and 2388
unique internal-compatible edge signatures. It can support 500 diverse edge instances without reuse:
`True`.

| environment | nodes | tools | parameters | graph edges | internal-compatible edges | max depth |
|---|---:|---:|---:|---:|---:|---:|
| Retail | 156 | 13 | 98 | 465 | 356 | 4 |
| GoogleTasks | 112 | 13 | 85 | 267 | 303 | 4 |
| Telecom | 103 | 12 | 66 | 336 | 250 | 4 |
| WhatsApp | 165 | 23 | 119 | 304 | 219 | 4 |
| Didi | 92 | 10 | 67 | 128 | 203 | 4 |
| Maps | 63 | 6 | 39 | 95 | 189 | 4 |
| TickTick | 108 | 10 | 69 | 216 | 165 | 4 |
| Airline | 127 | 10 | 63 | 226 | 142 | 4 |
| GitHubServer | 340 | 32 | 282 | 597 | 134 | 4 |
| GoogleSheets | 83 | 13 | 69 | 194 | 116 | 4 |
| CampusCard | 65 | 6 | 42 | 148 | 101 | 4 |
| Notion | 221 | 25 | 163 | 272 | 100 | 4 |
| MeditationServer | 109 | 7 | 72 | 169 | 89 | 4 |
| SanvelloMentalHealthServer | 92 | 6 | 53 | 147 | 89 | 4 |
| TravelBooking | 160 | 18 | 132 | 218 | 85 | 4 |
| Weather | 78 | 9 | 56 | 100 | 78 | 3 |
| TicketManagementSystem | 139 | 14 | 117 | 205 | 60 | 4 |
| Message | 84 | 13 | 48 | 112 | 52 | 4 |
| ExcelServer | 54 | 7 | 46 | 78 | 35 | 3 |
| Posting | 125 | 17 | 98 | 177 | 33 | 2 |
| PriceComparison | 46 | 5 | 39 | 57 | 33 | 3 |
| Dice | 84 | 5 | 63 | 110 | 30 | 4 |
| ResendEmailService | 56 | 5 | 40 | 78 | 28 | 4 |
| LinkedInJobs | 66 | 5 | 45 | 90 | 26 | 3 |
| HotelBooking | 57 | 5 | 39 | 65 | 25 | 4 |
| Canvas | 60 | 8 | 37 | 68 | 23 | 4 |
| FakeStoreServer | 84 | 8 | 53 | 105 | 22 | 4 |
| TradingBot | 127 | 20 | 94 | 150 | 22 | 3 |
| OneDrive | 67 | 6 | 53 | 87 | 21 | 4 |
| CoresignalJobs | 87 | 5 | 67 | 118 | 20 | 3 |
| ShopifyEcommerce | 100 | 5 | 50 | 114 | 19 | 3 |
| PostmarkEmailService | 88 | 5 | 70 | 98 | 18 | 4 |
| ZoomMeetingServer | 63 | 8 | 49 | 84 | 18 | 3 |
| AirtableMcpServer | 45 | 5 | 34 | 51 | 17 | 4 |
| StripePaymentServer | 77 | 5 | 44 | 93 | 17 | 4 |
| EbayServer | 89 | 5 | 52 | 104 | 16 | 4 |
| EtsyServer | 79 | 5 | 62 | 92 | 15 | 2 |
| PayPalPaymentProcessor | 55 | 5 | 41 | 73 | 15 | 4 |
| Calendar | 50 | 6 | 38 | 61 | 13 | 3 |
| ChinaRailway | 30 | 5 | 25 | 36 | 11 | 2 |
| FatSecretPlatform | 83 | 6 | 75 | 88 | 10 | 2 |
| GoogleDrive | 61 | 5 | 48 | 74 | 9 | 3 |
| KospiKosdaqStock | 38 | 4 | 33 | 49 | 9 | 2 |
| VehicleControl | 127 | 30 | 97 | 116 | 9 | 2 |
| GithubTrending | 50 | 2 | 22 | 49 | 8 | 4 |
| FinancialDatasets | 83 | 10 | 70 | 140 | 6 | 1 |
| GameTrending | 41 | 6 | 34 | 45 | 6 | 1 |
| HKBus | 38 | 7 | 31 | 50 | 6 | 1 |
| UUPaoTui | 30 | 5 | 25 | 34 | 6 | 2 |
| Filesystem | 68 | 11 | 52 | 72 | 5 | 3 |
| HowToCook | 37 | 5 | 31 | 37 | 5 | 2 |
| LeagueOfLegends | 138 | 15 | 117 | 135 | 5 | 2 |
| BestBuyServer | 80 | 5 | 72 | 86 | 4 | 2 |
| HackerNews | 53 | 5 | 43 | 53 | 4 | 2 |
| ClinicalTrialsGov | 44 | 5 | 35 | 47 | 3 | 2 |
| DrugBank | 48 | 5 | 43 | 50 | 3 | 1 |
| SlackServer | 58 | 5 | 50 | 61 | 3 | 2 |
| AmazonProduct | 52 | 5 | 42 | 57 | 2 | 2 |
| Kuaidi100 | 24 | 3 | 21 | 24 | 2 | 1 |
| MailgunCommunication | 59 | 5 | 48 | 66 | 2 | 1 |
| MovieRecommender | 29 | 3 | 26 | 29 | 2 | 1 |
| OpenLibrary | 46 | 5 | 41 | 47 | 2 | 1 |
| Valorant | 44 | 6 | 37 | 43 | 2 | 2 |
| ZillowRealEstate | 44 | 4 | 33 | 54 | 2 | 1 |
| AirDNA | 43 | 5 | 32 | 42 | 1 | 1 |
| GitServer | 84 | 14 | 63 | 89 | 1 | 2 |
| MetMuseum | 25 | 3 | 18 | 25 | 1 | 1 |
| PubMedServer | 40 | 5 | 35 | 39 | 1 | 1 |
| AdobePDFServices | 40 | 5 | 35 | 40 | 0 | 0 |
| AmazonSES | 32 | 6 | 25 | 32 | 0 | 0 |
| AttomRealEstate | 52 | 5 | 39 | 71 | 0 | 0 |
| Calculator | 5 | 1 | 4 | 4 | 0 | 0 |
| CarPrice | 15 | 3 | 12 | 14 | 0 | 0 |
| CryptoPrice | 26 | 3 | 23 | 25 | 0 | 0 |
| Dictionary | 29 | 5 | 24 | 28 | 0 | 0 |
| FruityviceServer | 11 | 2 | 9 | 10 | 0 | 0 |
| GorillaFileSystem | 55 | 18 | 36 | 50 | 0 | 0 |
| HugeiconsServer | 21 | 3 | 18 | 20 | 0 | 0 |
| MemoryKnowledgeGraph | 29 | 5 | 24 | 28 | 0 | 0 |
| MongoDBServer | 37 | 5 | 32 | 36 | 0 | 0 |
| NationalParks | 62 | 5 | 57 | 61 | 0 | 0 |
| RentCast | 60 | 5 | 45 | 80 | 0 | 0 |
| SimpleArxiv | 24 | 3 | 21 | 24 | 0 | 0 |
| TFTServer | 36 | 6 | 30 | 37 | 0 | 0 |
| Whois | 35 | 4 | 31 | 37 | 0 | 0 |
| WikipediaServer | 46 | 5 | 41 | 47 | 0 | 0 |
| WuWa | 9 | 3 | 6 | 9 | 0 | 0 |
| YahooFinance | 84 | 12 | 68 | 82 | 0 | 0 |

## B. Generation funnel

- sampled structures: 768
- query generated: 290
- static pass: 290
- MCP pass: 290
- replay pass: 289
- no-leak pass: 289
- final valid tasks: 289

## C. Structural result

- valid tasks: 289
- total gold edges: 528
- internal required edges: 528
- depth1 edges: 120
- depth2 edges: 198
- depth3+ edges: 210
- unique internal edge signatures: 470
- unique producer tools: 155
- unique consumer tools: 165
- environments: 44

## D. Drop reasons

- query_generation_failure: 478
- replay_failure: 1

## E. Frozen300 isolation

- exact overlap in retained pool: 0
- method: post-hoc one-way normalized-query hash comparison only
- Frozen query text, graph, trajectory, and state were not persisted or used for generation.

## F. Preference forecast

Using the prior observed conversion only as a forecast (33 edges -> 9 states -> 18 pairs):

- expected unique paired frontier states: about 144
- expected preference pairs: about 288

These are estimates, not actual Dynamic-v1 sampling results.

## Protocol stop

No Dynamic-v1 sampling, DPO, SFT, GRPO, PPO, full RL, Frozen300 evaluation, or BFCL was started.
