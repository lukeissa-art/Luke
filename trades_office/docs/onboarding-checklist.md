# Onboarding Checklist: The 2-Hour Setup

Every shop gets this, in person or on a video call. The answers go straight into the dashboard's **New shop** form. The fields in that form match the sections below.

## Before the call (15 min, you alone)

- [ ] Look up the shop: Google Business profile, website, reviews, service area, hours
- [ ] Pre-fill what you can in **New shop** (name, trade, hours, services from their website)
- [ ] Buy a local Twilio number in their area code and paste it into **Assistant phone number**
- [ ] Set its Voice, status and Messaging webhooks (they're listed at the bottom of the shop's dashboard page)

## Part 1: Learn the shop (45 min, with the owner)

**Basics**
- [ ] Owner's name and **cell number** (for transfers and text summaries)
- [ ] Time zone
- [ ] Business hours, day by day (include Saturday if they work it)

**Service area**
- [ ] Towns, zip codes or radius. Anywhere they *won't* go?

**Services**
- [ ] What they do (repair, install, replace, drain cleaning, maintenance agreements…)
- [ ] What they **don't** do (septic, wells, commercial, certain brands, mobile homes…). This goes in **Extra instructions**.

**Pricing the assistant may say**
- [ ] Service call or diagnostic fee? Waived with repair?
- [ ] After-hours or emergency rate?
- [ ] Anything else they're happy for it to quote. If they aren't comfortable, leave it blank and the assistant will say "the tech will give you an exact price on site."

**Emergencies**
- [ ] Walk through the built-in rules: gas smell, carbon monoxide and sparks/smoke get safety instructions and an immediate transfer; flooding, burst pipes, sewage, no heat, no AC and power outages get an URGENT text to the owner and an offer to transfer
- [ ] Anything else they treat as an emergency? (e.g. "walk-in freezer", "no hot water at a restaurant"). Add these to **Extra emergency keywords**.
- [ ] Will they actually answer transfers at 2am? If not, change the extra instructions to "take details and promise a call back first thing".

**Scheduling**
- [ ] Arrival window length (2 hours is standard)
- [ ] Do they want it to book straight into the calendar, or take messages only for now?
- [ ] Google Calendar? Have them share it with the service account email, then set **Calendar** to `google` and paste the calendar ID

**Follow-up**
- [ ] Google review link (Google Business Profile → "Ask for reviews")
- [ ] Average job value (for the ROI numbers in the weekly report)

## Part 2: Tune and test (45 min)

- [ ] Save the shop. Open **Test call in browser** and run these calls together:
  - [ ] Routine: "My kitchen sink is clogged. Can someone come tomorrow?" It should book and confirm the details.
  - [ ] Emergency: "I smell gas." It should give safety instructions and transfer.
  - [ ] Urgent: "Water is pouring through my ceiling." It should offer to transfer.
  - [ ] Out of area or a service they don't offer. It should take a message and not book.
  - [ ] Price question. It should say only what's in the pricing notes.
  - [ ] Salesperson or spam. It should end politely.
- [ ] Fix any wording in **Extra instructions** and re-test
- [ ] Call the real Twilio number from the owner's phone once, end to end

## Part 3: Turn it on (15 min)

- [ ] **Trial:** set up after-hours forwarding only (carrier "forward when unanswered" or a scheduled forward in their phone system). Keep **When to answer** set to "After hours + when owner misses a call".
- [ ] Show the owner the text commands: `DONE <appt#>`, `QUOTE <phone> <amount> <description>`, `PAUSE`, `RESUME`
- [ ] Tell them what to expect: a text for every call, a weekly report every Monday, and your cell for anything that feels off
- [ ] Put a reminder in your calendar for **day 14**: the trial review meeting

## After setup

- [ ] Day 1 to 7: read every transcript each morning and send the owner a quick "here's what we caught" text
- [ ] Day 14: trial review and close (see the [sales playbook](sales-playbook.md))
