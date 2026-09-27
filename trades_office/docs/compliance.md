# Compliance Notes

This isn't legal advice. Have a local attorney review your customer terms and these practices before launch. It's a small cost next to the risk.

## Call recording and AI disclosure

- The assistant opens every call with *"This call may be recorded. I'm the shop's virtual assistant."* (see `greeting()` in `app/receptionist.py`).
- Some states require **all-party consent** to record (for example California, Florida, Illinois, Maryland, Massachusetts, Montana, New Hampshire, Pennsylvania, Washington). The upfront disclosure is the standard way to handle this. Check your state and your customers' states.
- The app stores a **text transcript**, not audio. If you turn on Twilio call recording, the disclosure above still applies, and you should set a retention period.
- Some states have laws on disclosing AI or bot use in calls. The greeting discloses it, and the assistant is told to say it's a virtual assistant if asked.

## Texting (SMS)

- **A2P 10DLC registration is required** for business texting from US local numbers. Register your brand and a campaign in the Twilio console before sending follow-ups. Unregistered traffic gets filtered or blocked. Approval can take 1 to 3 weeks, so start in week 1.
- **Consent:** texts go only to people who called the shop or are existing customers. Those are transactional or relationship messages. Keep marketing blasts out of this product.
- **Opt-out:** every customer follow-up text includes "Reply STOP to opt out." STOP, UNSUBSCRIBE and similar words are honored immediately: the number is saved in `opt_outs` and all pending texts to it are cancelled. START re-subscribes.
- **Quiet hours:** automated customer texts are never sent between 8pm and 8am in the shop's local time (`app/followups.py`).
- Texts to the shop owner (summaries, urgent alerts, weekly reports) are service messages to your customer and are covered by your customer terms.

## Customer (shop) terms: what to include

- Service description, plan limits (monthly call caps), and what happens over the cap (calls pass through to the owner)
- The owner is responsible for the accuracy of the information they give (prices, service area, hours)
- The assistant is not a substitute for emergency services; emergencies are transferred to the owner
- Month-to-month billing, a cancel-anytime policy, and the annual plan discount if you offer one
- Data: you store call transcripts, caller phone numbers and appointment details to provide the service, and don't sell them
- Limitation of liability, capped at fees paid

## Data handling

- Transcripts contain names, addresses and phone numbers. Keep the database on an encrypted volume and restrict dashboard access (a long `ADMIN_PASSWORD`, HTTPS only).
- Twilio webhook signatures are validated when Twilio credentials are set (`TWILIO_VALIDATE_SIGNATURES=true`). Leave it on in production.
- Stripe webhook signatures are always validated.
