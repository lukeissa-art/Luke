# Put TradeDesk AI live

The whole product (website, dashboard, phone webhooks, follow-up jobs) runs as **one service** with one small database file on a persistent disk. Pick **one** host below. Both take about 15 minutes.

Have these ready:
- **Anthropic API key** (console.anthropic.com → API keys). Without it the site loads, but the assistant transfers every call to the owner.
- Your **cell number** (you get a text for every trial signup)
- A **contact email** for the website
- Optional for now: Twilio account SID and auth token, plus one Twilio number for company texts

---

## Option A: Render (easiest)

1. Go to **dashboard.render.com → New → Blueprint**.
2. Connect GitHub and pick the **lukeissa-art/Luke** repo. For the branch, pick `main` once this work is merged, or `claude/ai-trades-back-office-aoy266` until then.
3. Render reads `render.yaml` and shows one web service, **tradedesk-ai**, with a 1 GB disk. It asks for the values marked "sync: false":
   - `ANTHROPIC_API_KEY`, `CONTACT_EMAIL`, `FOUNDER_PHONE` (fill these in now)
   - `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `COMPANY_SMS_NUMBER` (leave blank until Twilio is set up; texts are logged instead of sent)
4. Click **Apply**. The first build takes 3 to 5 minutes. The plan is **Starter (~$7/month)**, because a persistent disk needs a paid instance.
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
   ANTHROPIC_API_KEY=<your key>
   CONTACT_EMAIL=<your email>
   FOUNDER_PHONE=<your cell, e.g. +15125550123>
   SEED_DEMO=true
   PUBLIC_DEMO_SHOP_ID=1
   ```
5. Deploy. Open the generated domain; the dashboard is at `/admin/`.

`PUBLIC_BASE_URL` is picked up automatically from Render and Railway. Set it yourself only once you add a custom domain.

---

## After it's live (10 minutes)

1. **Check the site:** fill in the trial form with your own details. It should appear at the top of `/admin/`, and if Twilio is set up you get a text.
2. **Try the demo** on the home page ("Be the customer"). If it only ever says it's connecting you to the team, the Anthropic key is missing or wrong. Check the service logs.
3. **Custom domain:** buy one (e.g. from Cloudflare or Namecheap), add it in Render (Settings → Custom Domains) or Railway (Networking → Custom Domain), create the DNS record they show you, then set `PUBLIC_BASE_URL=https://yourdomain.com` and redeploy.
4. **Real company details:** set `COMPANY_NAME`, `SERVICE_REGION` and `CONTACT_EMAIL` in the environment and redeploy.

## Turning on phone calls and texts (when you're ready for trial shops)

1. Create a Twilio account and upgrade it (trial accounts can only call verified numbers).
2. Start **A2P 10DLC** registration (Messaging → Regulatory Compliance). Use your live `/privacy` URL. Approval takes 1 to 3 weeks.
3. Set `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN` and `COMPANY_SMS_NUMBER` and redeploy.
4. For each shop, buy a number and set its webhooks as listed at the bottom of the shop's dashboard page.
5. Optional **demo line** for the website: buy one number, set its webhooks the same way, put that number into the demo shop's "Assistant phone number" in the dashboard, and set `DEMO_PHONE_NUMBER` so the home page shows it.

## Running costs to expect

- Hosting: about $5 to $7/month
- Claude: a few cents per call (check the Anthropic console after your first week)
- Twilio: about $1.15/month per number, plus about $0.01 to $0.02 per call minute and about $0.008 per text, plus A2P registration fees

## Backups

The database is one file at `/data/trades_office.db`. Once you have paying shops, download a copy weekly (in the host's shell: `python -c "import sqlite3; sqlite3.connect('/data/trades_office.db').backup(sqlite3.connect('/data/backup.db'))"`), or move to managed Postgres later.
