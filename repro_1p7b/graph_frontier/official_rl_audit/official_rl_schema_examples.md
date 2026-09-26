# Official EnvFactory-RL schema examples

50 rows from distinct environment combinations where possible; no raw state, secrets, or long query text.

| Row | Environment | Query chars | Gold calls | Args/task | Masked slots | Servers |
|---:|---|---:|---:|---:|---:|---|
| 0 | PubMedServer | 771 | 4 | 10 | 4 | PubMedServer |
| 3 | AirtableMcpServer | 367 | 2 | 1 | 0 | AirtableMcpServer |
| 14 | SlackServer | 535 | 5 | 9 | 4 | SlackServer |
| 19 | ClinicalTrialsGov | 1555 | 12 | 24 | 5 | ClinicalTrialsGov |
| 27 | Whois | 532 | 1 | 1 | 0 | Whois |
| 30 | NationalParks | 474 | 2 | 4 | 2 | NationalParks |
| 35 | WikipediaServer | 1249 | 3 | 5 | 2 | WikipediaServer |
| 38 | OpenLibrary | 427 | 1 | 3 | 0 | OpenLibrary |
| 61 | MemoryKnowledgeGraph | 413 | 2 | 2 | 0 | MemoryKnowledgeGraph |
| 81 | MongoDBServer | 691 | 7 | 20 | 5 | MongoDBServer |
| 387 | Notion | 296 | 1 | 4 | 1 | Notion |
| 388 | HackerNews,Posting | 258 | 3 | 6 | 3 | HackerNews,Posting |
| 391 | TickTick | 478 | 3 | 2 | 0 | TickTick |
| 392 | Canvas,GoogleTasks | 304 | 5 | 6 | 0 | GoogleTasks,Canvas |
| 393 | GoogleTasks,TickTick | 246 | 3 | 6 | 1 | GoogleTasks,TickTick |
| 395 | ResendEmailService | 122 | 1 | 0 | 0 | ResendEmailService |
| 396 | PostmarkEmailService,ResendEmailService | 482 | 7 | 24 | 6 | PostmarkEmailService,ResendEmailService |
| 398 | GorillaFileSystem | 117 | 2 | 1 | 1 | GorillaFileSystem |
| 400 | Filesystem,GorillaFileSystem | 232 | 2 | 3 | 0 | GorillaFileSystem,Filesystem |
| 404 | PayPalPaymentProcessor | 279 | 1 | 4 | 1 | PayPalPaymentProcessor |
| 405 | EtsyServer,StripePaymentServer | 281 | 2 | 3 | 1 | EtsyServer,StripePaymentServer |
| 406 | Didi,Weather | 235 | 2 | 2 | 1 | Weather,Didi |
| 407 | Weather | 277 | 2 | 1 | 0 | Weather |
| 408 | WhatsApp | 264 | 2 | 1 | 0 | WhatsApp |
| 410 | Didi,Maps | 285 | 3 | 7 | 2 | Didi,Maps |
| 411 | ChinaRailway | 232 | 1 | 1 | 0 | ChinaRailway |
| 413 | Telecom | 236 | 5 | 8 | 0 | Telecom |
| 415 | Retail | 285 | 3 | 3 | 0 | Retail |
| 419 | CampusCard | 169 | 1 | 1 | 0 | CampusCard |
| 420 | TradingBot | 77 | 1 | 0 | 0 | TradingBot |
| 423 | Kuaidi100 | 104 | 1 | 1 | 0 | Kuaidi100 |
| 431 | Airline | 372 | 4 | 15 | 1 | Airline |
| 437 | GameTrending | 170 | 2 | 2 | 2 | GameTrending |
| 439 | FakeStoreServer | 360 | 2 | 1 | 0 | FakeStoreServer |
| 441 | EbayServer,EtsyServer | 523 | 4 | 17 | 5 | EtsyServer,EbayServer |
| 443 | HotelBooking | 197 | 2 | 4 | 0 | HotelBooking |
| 453 | Message | 308 | 5 | 5 | 2 | Message |
| 456 | GitHubServer | 633 | 3 | 3 | 0 | GitHubServer |
| 457 | Didi,Maps,Weather | 365 | 5 | 5 | 0 | Weather,Didi,Maps |
| 458 | Didi | 311 | 2 | 5 | 0 | Didi |
| 461 | Message,WhatsApp | 584 | 2 | 3 | 2 | Message,WhatsApp |
| 464 | MeditationServer,Message | 277 | 5 | 3 | 0 | MeditationServer,Message |
| 467 | MeditationServer | 318 | 1 | 2 | 0 | MeditationServer |
| 468 | Dice,LinkedInJobs | 322 | 6 | 10 | 1 | LinkedInJobs,Dice |
| 469 | LinkedInJobs | 510 | 3 | 3 | 0 | LinkedInJobs |
| 470 | Dice | 554 | 3 | 6 | 1 | Dice |
| 471 | Canvas | 394 | 4 | 3 | 0 | Canvas |
| 472 | GoogleTasks | 446 | 6 | 7 | 14 | GoogleTasks |
| 474 | UUPaoTui | 191 | 2 | 1 | 0 | UUPaoTui |
| 480 | AirDNA | 397 | 4 | 6 | 0 | AirDNA |

The top-level release contains no independent task ID, seed, tool observations, trajectory/history, or assistant final response. ground_truth is a JSON-encoded ordered list of tool calls with names, arguments and masked_arguments. Masked fields are present in JSON but excluded from official matching; their value fidelity is not independently proven.
