# Operations

## Daily (15 min)

- Open the dashboard overview and check that every shop is taking calls. A shop at zero calls for 2+ days usually means forwarding broke.
- Check the worker is running: follow-up texts on shop pages should move from `pending` to `sent`.
- Read transcripts for any call marked **urgent** or **transferred**, plus every call from shops in their first week.

## Weekly (1 hour)

- **Transcript review:** read 5 random calls per shop. Look for wrong information, awkward loops (asking the same thing twice), missed bookings, and price quotes that weren't approved. Fix them in the shop's **Extra instructions** and re-test with **Test call in browser**.
- Weekly reports go out automatically at 8am Central on Mondays. Call any owner whose numbers dropped.
- Check costs: the Anthropic console (usage by day) and the Twilio console (voice minutes, SMS). The target is ≤ $90 per shop per month. If it's high, look for long calls first. They usually come from spam or loops.

## Monthly

- Check call counts against plan caps (the shop page shows "calls this month"). A shop near its cap is an upgrade conversation, not a cutoff. Over the cap, calls pass straight through to the owner.
- Trials ending this week: book the review meeting.
- Churn review: for every cancellation, write down why.

## Support playbook

| Owner says | Check |
|---|---|
| "It's not answering." | Is the shop status `trial` or `active`? Is forwarding still set on their phone? Are the Twilio number's webhooks correct? |
| "It booked something wrong." | Read the transcript. If the hours are wrong, edit the schedule. If it misheard the address, that's speech recognition; add "spell the street name if unsure" to extra instructions. |
| "It told someone a price." | Check the pricing notes. Remove anything the owner doesn't want said. |
| "Turn it off right now." | They can text `PAUSE`, or set the status to `paused` in the dashboard. Calls then ring straight through to their cell. |
| "A customer didn't get the text." | Did the customer reply STOP? (Check the `opt_outs` table.) Is A2P 10DLC registration approved? |

## Incidents

If the assistant mishandles an emergency call:
1. Pause the shop (`paused`) until it's fixed. Calls go straight to the owner.
2. Call the owner the same day. Explain what happened and what you changed.
3. Add a regression test in `tests/test_emergency.py` for the exact wording the caller used.

## Metrics that matter

- **Calls answered per shop per week.** Below 10 means forwarding is probably misconfigured.
- **Booked or clean-message rate.** Target 80%+.
- **Emergency transfer accuracy.** Must be 100%.
- **Trial → paid.** Target 50%+.
- **Monthly churn.** Target under 3%.
- **Cost to serve.** Target $50 to $90 per shop per month.
