# 90-Day Launch Plan

The goal is to prove owners will pay before real money is spent. There are three phases, each ending in a go/no-go gate. If a gate is missed, fix the offer or switch trade or task before moving on. Don't push through.

Budget for the 90 days: about $5k of the $20k startup money (LLC and insurance, Twilio and Claude usage, demo line, gas). The rest stays in reserve until Gate 2 passes.

---

## Phase 1: Validate the problem (days 1 to 30)

**Goal:** 20 real conversations with HVAC and plumbing owners before selling anything.

| Week | Do this |
|---|---|
| 1 | Form the LLC, get general liability and E&O insurance quotes, open a business bank account. Pick the region (a 45-minute drive radius). List 80 target shops from Google Maps: 2 to 15 employees, reviews that mention "couldn't reach them" or "never called back" are gold. |
| 1 | Deploy this app, buy a demo number, and set up the demo shop (`python -m scripts.seed_demo`). Call it yourself 20 times with real scenarios: leak, no heat, gas smell, price shopper, spam. Fix the script until you'd trust it with your own shop. |
| 2–3 | Visit shops early (7 to 8am, before trucks roll) and supply houses (counter guys know everyone). Use the discovery questions in the [sales playbook](sales-playbook.md). **Don't pitch.** Log every answer in a simple sheet. |
| 3 | Go to one trade association meeting (PHCC, ACCA, or the local contractor association). |
| 4 | Tally the results. Count how many owners named missed calls or slow quotes unprompted, and write down their exact words. You'll use them in the pitch. |

**Gate 1 (day 30):** At least **10 of 20** owners describe missed calls or slow quotes as a real, costly problem, and at least **5** agree to a free trial.
- *Missed:* try the same questions with electrical shops, or pivot the offer toward quote follow-up if that's what they complained about. Don't build more.

---

## Phase 2: Free trials (days 31 to 60)

**Goal:** 5 to 8 shops on the 14-day after-hours trial; real calls answered; the assistant tuned on real transcripts.

| Week | Do this |
|---|---|
| 5 | Run the [onboarding checklist](onboarding-checklist.md) with each trial shop (2 hours each). Start them on **after-hours forwarding only**, so there's zero risk to their daytime business. |
| 5–8 | **Read every transcript daily** for the first week of each trial. Fix the shop's instructions on the same day you find a problem. Text the owner a quick "here's what we caught last night" each morning. That habit is the product. |
| 6 | Twilio A2P 10DLC registration should be approved by now. Turn on follow-up texts and review requests. |
| 7–8 | At day 14 of each trial, sit down with the owner and walk through the dashboard: calls caught, after-hours calls, jobs booked, estimated revenue. Then ask for the sale (see "Trial close" in the playbook). |

**Gate 2 (day 60):** At least **3 shops paying** (setup fee waived for the first 10), the assistant booked real jobs, and **no safety incident** (every emergency was transferred correctly).
- *Missed because the product failed:* fix it and extend trials by two weeks.
- *Missed because owners won't pay:* test Starter at $249, or lead with the after-hours-only offer. Talk to the owners who said no and learn why.

---

## Phase 3: First revenue and repeatability (days 61 to 90)

**Goal:** 10 paying shops, 2 written case studies, and a pitch that works without you improvising.

| Week | Do this |
|---|---|
| 9 | Ask each paying owner for one referral (the incentive is one free month per referred shop that signs). Ask your 3 best for a testimonial and permission to use their numbers. |
| 9–10 | Write 2 one-page case studies from real dashboard numbers ("Caught 31 after-hours calls and booked $9,400 in jobs in one month"). |
| 10 | Pitch 2 supply houses on a partnership: they hand out your flyer and you mention them in your onboarding. Offer them a referral fee or co-branded lunch-and-learns. |
| 11 | Move the 24/7 answering upsell (Pro) to trial shops that are happy on after-hours. |
| 12 | Review the numbers against the targets below and write down the pitch, the objections you hear most, and the setup steps that still take too long. |

**Gate 3 (day 90):** **10 paying shops**, churn of zero or one, average price **≥ $350/month**, cost to serve **≤ $90/shop/month** (check your Claude and Twilio bills), and each setup taking **≤ 2 hours**.
- *Met:* spend the reserved startup money on the integrations contractor (Jobber / Housecall Pro) and a second region or trade.
- *Missed:* stay at 10 shops and fix what's broken before adding more.

---

## Metrics to track weekly (from the dashboard)

| Metric | Target |
|---|---|
| Owner conversations / trials started / paying | 20 / 5+ / 3+ by day 60 |
| Calls answered per shop per week | 20+ |
| Share of calls that end booked or with a clean message | 80%+ |
| Emergencies transferred correctly | 100% |
| Trial → paid conversion | 50%+ |
| Cost to serve per shop | ≤ $90/month |
