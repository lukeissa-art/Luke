# Put TradeDesk AI live

The whole product (website, dashboard, phone webhooks, follow-up jobs) runs as **one service** with one small database file on a persistent disk. Pick **one** host below. Both take about 15 minutes.

Have these ready:
- **An AI key**: either a **Gemini API key** (aistudio.google.com/apikey → Create API key; there's a free tier) or an **Anthropic API key** (console.anthropic.com → API keys). Without one the site loads, but the assistant transfers every call to the owner.
- Your **cell number** (you get a text for every trial signup)
- A **contact email** for the website
- Optional for now: Twilio account SID and auth token, plus one Twilio number for company texts

---

## Option A: Render (easiest)

1. Go to **dashboard.render.com → New → Blueprint**.
2. Connect GitHub and pick the **lukeissa-art/Luke** repo, branch `main`.
3. Render reads `render.yaml` and shows one web service, **tradedesk-ai**, on the **Free** plan. It asks for the values marked "sync: false":
   - `GEMINI_API_KEY` **or** `ANTHROPIC_API_KEY` (one is enough; leave the other blank), plus `CONTACT_EMAIL` and `FOUNDER_PHONE` (fill these in now)
   - `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `COMPANY_SMS_NUMBER` (leave blank until Twilio is set up; texts are logged instead of sent)
4. Click **Apply**. The first build takes 3 to 5 minutes.

   **About the free plan:** the service sleeps after about 15 minutes with no visitors, and the first visit after that takes up to a minute to load. There is no disk, so trial signups and shops you add are **wiped on every restart or deploy** (the demo shop is recreated automatically). That's fine for showing the website. Before real trial shops or phone calls, upgrade: in `render.yaml`, change `plan: free` to `plan: starter` (~$7/month) and uncomment the `disk` block, or in the Render dashboard change the instance type to Starter and add a 1 GB disk mounted at `/data`.
5. When it shows **Live**, open `https://tradedesk-ai.onrender.com` (Render shows the exact URL).
6. Your dashboard login: the username is `admin`, and the password is under the service's **Environment** tab (`ADMIN_PASSWORD`, generated for you). Open `/admin/`.

## Option B: Railway (you already use it for the trading bot)

1. In your Railway project: **New → GitHub Repo → lukeissa-art/Luke**.
2. Open the new service → **Settings**:
   - **Root Directory:** `trades_office` (Railway then uses `trades_office/railway.toml` and the Dockerfile)
   - **Branch:** `main` once merged, or `claude/ai-trades-back-office-aoy266`
   - **Networking → Generate Domain**
3. **Volumes → New Volume**, attached to this service, mount path `/data`.
4. **Variables:** add
   ```
   ADMIN_PASSWORD=<a long password you choose>
   GEMINI_API_KEY=<your key>        # or ANTHROPIC_API_KEY=<your key> to use Claude
   CONTACT_EMAIL=<your email>
   FOUNDER_PHONE=<your cell, e.g. +15125550123>
   SEED_DEMO=true
   PUBLIC_DEMO_SHOP_ID=1
   ```
5. Deploy. Open the generated domain; the dashboard is at `/admin/`.

`PUBLIC_BASE_URL` is picked up automatically from Render and Railway. Set it yourself only once you add a custom domain.

---

## Switching the AI model on a running service

In Render (or Railway), open the service → **Environment**, add `GEMINI_API_KEY` with your key, and save. The service restarts and uses Gemini from then on. To go back to Claude, delete `GEMINI_API_KEY` (or set `LLM_PROVIDER=anthropic`).

## After it's live (10 minutes)

1. **Check the site:** fill in the trial form with your own details. It should appear at the top of `/admin/`, and if Twilio is set up you get a text.
2. **Try the demo** on the home page ("Be the customer"). If it only ever says it's connecting you to the team, the AI key is missing or wrong. Check the service logs.
3. **Custom domain:** buy one (e.g. from Cloudflare or Namecheap), add it in Render (Settings → Custom Domains) or Railway (Networking → Custom Domain), create the DNS record they show you, then set `PUBLIC_BASE_URL=https://yourdomain.com` and redeploy.
4. **Real company details:** set `COMPANY_NAME`, `SERVICE_REGION` and `CONTACT_EMAIL` in the environment and redeploy.

## Turning on phone calls and texts (when you're ready for trial shops)

1. Create a Twilio account and upgrade it (trial accounts can only call verified numbers).
2. Start **A2P 10DLC** registration (Messaging → Regulatory Compliance). Use your live `/privacy` URL. Approval takes 1 to 3 weeks.
3. Set `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN` and `COMPANY_SMS_NUMBER` and redeploy.
4. For each shop, buy a number and set its webhooks as listed at the bottom of the shop's dashboard page.
5. Optional **demo line** for the website: buy one number, set its webhooks the same way, put that number into the demo shop's "Assistant phone number" in the dashboard, and set `DEMO_PHONE_NUMBER` so the home page shows it.

## Turning on Google Calendar (free, one time, about 10 minutes)

This lets each shop owner block off time in their own Google Calendar and see every job the assistant books. You set it up once for the whole company; each shop then just shares their calendar.

1. Go to **console.cloud.google.com** (any Google account works) and create a project, e.g. "TradeDesk Calendar".
2. **APIs & Services → Library**, search **Google Calendar API**, click **Enable**.
3. **IAM & Admin → Service Accounts → Create service account**. Name it `tradedesk-calendar`, click **Done** (no roles needed).
4. Click the new service account → **Keys → Add key → Create new key → JSON**. A file downloads.
5. Open that file in a text editor, copy everything from `{` to `}`, and add it in Railway (or Render) → **Variables** as `GOOGLE_SERVICE_ACCOUNT_JSON`. Save; the app restarts.

Then on each shop's dashboard page, the **Google Calendar** card shows the email to share with and has a **Connect & test** button. The Calendar API is free at this volume. Keep the key file private, like a password.

## Running costs to expect

- Hosting: about $5 to $7/month
- AI model: a few cents per call or less (Gemini has a free tier with rate limits; check usage in Google AI Studio or the Anthropic console after your first week)
- Twilio: about $1.15/month per number, plus about $0.01 to $0.02 per call minute and about $0.008 per text, plus A2P registration fees

## Backups

The database is one file at `/data/trades_office.db`. Once you have paying shops, download a copy weekly (in the host's shell: `python -c "import sqlite3; sqlite3.connect('/data/trades_office.db').backup(sqlite3.connect('/data/backup.db'))"`), or move to managed Postgres later.
