# Business Plan: AI Back Office for Trades

Sep 27, 2026 · Luke Issa

> All figures in this plan are working assumptions to be tested in the first 90 days, not researched market data.

## Executive summary

The company sells an AI front office to small plumbing, HVAC and electrical shops. It answers every call, books jobs, writes quotes and follows up, for a flat monthly fee. The goal for year one is 60 paying shops at about $400 a month, which is roughly $290k in annual recurring revenue.

Small trades shops lose jobs every day to missed calls and slow quotes, because the owner is on a ladder, not at a desk. Our product catches those jobs using existing AI models, so we build workflows and relationships, not technology from scratch. We start with one trade in one region, win on setup help and local service, then expand trade by trade.

## The problem

A small trades shop typically misses a large share of inbound calls, and most callers who reach voicemail call the next company instead. One missed emergency plumbing or HVAC call can mean a lost $300 to $1,500 job.

- **Missed calls.** The owner and techs are on job sites. Nobody answers during working hours, evenings or weekends.
- **Slow quotes.** Estimates get written at night from memory or scribbled notes, often days late.
- **No follow-up.** Open quotes, maintenance reminders and review requests fall through the cracks.
- **Admin overload.** A full-time office person costs $35k to $50k a year, which a 2 to 8 person shop often can't justify.

Existing field-service software schedules and invoices jobs once they exist. It does little to capture the job in the first place, which is where the money leaks.

## The product

The first version does three jobs well: answer, book and follow up. Everything else waits until 20 shops are paying.

**MVP (launch in 60 days)**. Built: see the [README](../README.md).
- **AI receptionist.** Answers calls 24/7 in the shop's name, triages emergencies, collects address and problem details, and texts the owner a summary.
- **Booking.** Offers open slots from the shop's calendar or field-service software and confirms by text.
- **Follow-up.** Automatic texts for unbooked callers, open quotes, and Google review requests after each job.

**Later (months 4 to 12)**
- **Quote writer.** Turns a tech's photos and voice notes into a clean, priced estimate using the shop's own price book.
- **Maintenance plans.** Reminders for seasonal HVAC tune-ups and water heater flushes, which create repeat revenue for the shop. *(Reminder texts are built and on the Plus plan.)*
- **Weekly report.** Calls answered, jobs booked and estimated revenue captured, so the owner sees the product paying for itself. *(Built.)*

## Target customer and market

The beachhead is owner-operated HVAC and plumbing shops with 2 to 15 people within driving distance. These trades have the most emergency calls, the highest job values, and seasonal rushes when missed calls hurt most.

**Ideal customer**
- The owner still answers the phone, or relies on a spouse or part-time helper
- 20 to 150 inbound calls a week, $300k to $3M in yearly revenue
- Already paying for Google ads or a directory listing, so they know what a lead costs
- Uses a calendar or a basic field-service app, or nothing at all

**Market size (approximate).** The US has well over 200,000 plumbing, HVAC and electrical contracting businesses, most of them small. At $400 a month, capturing just 1% of them would be roughly $10M in annual revenue. The bigger point is that the market is large enough for a regional business to grow for years without running out of customers.

## Competition and positioning

We win by being the local, done-for-you option built only for trades, not by having better AI. The technology is widely available, so service and focus are the moat.

| Competitor type | Examples | Their weakness for our customer |
|---|---|---|
| Field-service software | ServiceTitan, Jobber, Housecall Pro | Built to manage jobs, not capture them; add-on AI features are generic and self-serve |
| Human answering services | Ruby, Smith.ai, local call centers | Per-minute pricing gets expensive; operators don't know the trade or the price book |
| Generic AI receptionists | Many venture-funded startups | Self-serve setup, no local presence, not tuned to emergencies or trade pricing |
| Doing nothing | Voicemail, owner's cell phone | Free, but silently loses jobs every week |

**Our position:** "We answer, book and follow up for HVAC and plumbing shops, and we set it all up for you." Every customer gets a hands-on setup call, a script tuned to their services, and a real person to call when something's off.

> AI receptionist tools have multiplied since 2024. Check current competitors and pricing in your region before finalizing prices.

## Pricing and revenue model

Revenue is a flat monthly subscription plus a one-time setup fee, averaging about $400 a month per shop. Flat pricing is easy for owners to say yes to: one booked job a month pays for it.

| Plan | Price per month | Includes |
|---|---|---|
| Starter | $249 | After-hours and overflow call answering, text summaries, up to 150 calls |
| Pro | $449 | 24/7 answering, booking into calendar, follow-up texts, review requests, up to 400 calls |
| Plus | $799 | Everything in Pro, plus quote writer, maintenance-plan reminders, multiple locations |

**Setup fee:** $500, waived for the first 10 customers in exchange for a testimonial.

**Unit economics (targets)**
- Cost to serve: $50 to $90 per shop per month (AI usage, phone numbers, texting, software)
- Gross margin: about 80%
- Cost to acquire a customer: under $1,000, mostly the founder's time early on
- Payback: 3 months or less
- Churn: under 3% of customers per month; owners who see booked jobs rarely cancel

## Go-to-market

The first customers come from showing up in person. Ads come later, once we know what message converts. Trades owners buy from people they trust, and a live demo of the AI answering their own calls is the strongest pitch. Details are in the [sales playbook](sales-playbook.md).

**Customers 1 to 10: founder-led, local**
- Visit 50 shops, supply houses and trade association meetings. Ask what eats their time before pitching.
- Offer a free 14-day trial that forwards only after-hours calls, so there's no risk to their business.
- At the end of the trial, show each owner the calls it caught and the jobs it booked.

**Customers 11 to 50: referrals and partners**
- Give a referral credit of one free month for every shop a customer refers.
- Partner with plumbing and HVAC supply houses, equipment distributors and trade-business coaches.
- Publish short case studies: "Caught 31 after-hours calls and booked $9,400 in jobs in one month."

**Customers 51 to 200: repeatable channels**
- Local Google and Facebook ads aimed at shop owners, using proven case-study numbers.
- Expand to a second trade (electrical) and a second region.
- Hire the first salesperson once the founder has closed 40 or more deals and the pitch is written down.

## Operations and tech stack

Buy nearly everything and build only the trade-specific glue.

| Layer | Approach | Monthly cost at 50 shops (approx.) |
|---|---|---|
| Voice AI | Claude for the conversation, Twilio speech recognition and text-to-speech | $1,500 to $3,000 |
| Phone and texting | Twilio for numbers, call forwarding and SMS | $300 to $600 |
| Integrations | Google Calendar first, then Jobber and Housecall Pro | Developer time |
| Customer records | Built-in dashboard (shops, trials, calls, MRR) | $0 |
| Billing | Stripe subscriptions | Percentage of revenue |

**Team**
- Year 1: the founder handles sales, setup and support, with one part-time contractor for integrations
- Year 2: first salesperson, plus one support and onboarding person
- Year 3: a second salesperson, a full-time engineer and a support lead

Each new shop takes about 2 hours to set up. See the [onboarding checklist](onboarding-checklist.md).

## Financial projections

The business can start for about $20k and reach break-even, including a modest founder salary, at around 30 shops in months 8 to 10. By year three it projects about $1.9M in revenue and $680k in profit.

**Startup costs: about $20,000**
- LLC, insurance, contracts: $2,000
- Contractor to build first integrations and call flows: $10,000 *(the MVP in this repo covers much of this)*
- Software and phone tools for the first 3 months: $3,000
- Sales materials, demo line, local events: $2,000
- Cash reserve: $3,000

| Year | Shops at year end | Avg. price per month | Revenue | Total costs | Profit |
|---|---|---|---|---|---|
| 1 | 60 | $400 | $169k | $125k | $44k |
| 2 | 200 | $420 | $725k | $459k | $266k |
| 3 | 450 | $450 | $1.88M | $1.20M | $677k |

Revenue is the average shop count across each year times price, plus $500 setup fees for new shops. Costs include cost to serve, salaries (a $50k founder salary in year 1), marketing and tools. The biggest swing factor is churn: at 5% a month instead of 3%, year-three profit falls by roughly a third.

## Risks and mitigations

The biggest risk is that field-service software vendors bundle a good-enough AI receptionist for free. Our defense is service, trade focus and local relationships, which big vendors don't offer small shops.

| Risk | Mitigation |
|---|---|
| Big software vendors bundle AI answering | Integrate with them rather than compete; sell hands-on setup and results, not features |
| The AI mishandles an emergency or angers a customer | Hard-coded emergency rules (gas smell, flooding) transfer to the owner immediately; any AI failure transfers too; review call transcripts weekly |
| Owners churn after the busy season | Annual plans at a discount; maintenance-plan reminders keep value flowing in slow months |
| AI and phone costs rise | Flat pricing with call caps per plan; switch AI models if one gets expensive |
| Call recording and consent laws | Disclosure at the start of every call; check your state's recording-consent rules; use standard customer terms ([compliance](compliance.md)) |
| Founder burnout from support load | Standard setup checklist and scripts; hire support at around 80 shops |

## 90-day launch plan

The first 90 days prove that owners will pay before any real money is spent. Each phase ends with a gate. If a gate isn't met, fix the offer or switch trades before moving on. See [launch-plan-90-days.md](launch-plan-90-days.md).

> If fewer than 10 of 20 owners describe missed calls or slow quotes as a real problem, test the plan on a different trade or task before building anything.
