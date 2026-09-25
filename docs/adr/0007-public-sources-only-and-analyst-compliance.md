# ADR-0007: Public sources only; analyst-firm pages only when explicitly supplied

- Status: Accepted
- Date: 2026-09-24
- Deciders: Kehinde Ogunlowo

## Context

Account teams want analyst context (Gartner, Forrester) in client briefs. Analyst
research documents are licensed, paywalled content, and the firms' terms of use
prohibit automated scraping, crawling and systematic retrieval of their sites.
SEC EDGAR is public and free but has a fair-access policy (declared User-Agent
with contact e-mail, at most 10 requests per second). Corporate sites publish
`robots.txt` that the crawler must honour. A crawler that follows arbitrary
links is also an SSRF vector into the workspace network.

## Decision

1. **Sources.** Evidence comes from:
   - SEC EDGAR JSON endpoints (`research/sources/edgar.py::EdgarClient`):
     `company_tickers.json`, `submissions/CIK##########.json`,
     `api/xbrl/companyfacts/...` with `companyconcept` fallback, and archive
     documents;
   - the company's own domain (`research/sources/corporate_site.py::CorporateSiteDiscoverer`):
     sitemaps declared in robots.txt plus `/sitemap.xml`, the homepage and one
     hop into hub pages (investor relations, newsroom, leadership);
   - analyst-firm public pages **only** when a human places the URL in
     `ResearchRequest.seed_urls` (`research/sources/analyst_public.py::PublicAnalystSource`).
2. **Analyst policy.** `PublicAnalystSource` never searches, crawls, follows
   links or reads sitemaps on analyst domains. It fetches exactly the supplied
   URLs, one request each, and only when they fall under a public path prefix
   (`DEFAULT_ANALYST_POLICIES`: `gartner.com` `/en/newsroom/`, `/en/articles/`;
   `forrester.com` `/press-newsroom/`, `/blogs/`). Paths that look like licensed
   research or account pages (`/document/`, `/doc/`, `/login`, `/account`,
   `/reprints` ...) are refused with `CrawlPolicyViolationError`. Such documents
   are typed `ANALYST_PUBLIC` with trust 0.7 (`TRUST_ANALYST`), below SEC (0.95)
   and first-party company pages (0.85).
3. **Gartner attribution in briefs.** The "Gartner-Relevant Insights" section maps
   evidence to publicly known Gartner strategic-technology *themes*
   (`briefing/opportunity.py::TREND_THEMES`). Every mapping is an
   `AI_RECOMMENDATION` carrying `MAPPING_NOTE`. Text that attributes a claim to
   Gartner ("Gartner predicts ...") survives only if it cites an `ANALYST_PUBLIC`
   evidence item that itself mentions Gartner.
4. **Crawl policy on every fetch** (`research/fetcher.py::PolicyEnforcingFetcher`):
   `UrlGuard` (SSRF and allow-list, re-checked on every redirect hop),
   `RobotsPolicy` (RFC 9309, fail-closed on 429/5xx/network error, `Crawl-delay`
   honoured), per-host token bucket `HostRateLimiter`
   (`crawler.requests_per_second_per_host`, default 1.0), retry with backoff and a
   per-host circuit breaker, response cap `crawler.max_response_bytes` checked on
   the decompressed stream.
5. **Identification.** Every request carries the declared User-Agent
   (`crawler.user_agent`) and a `From` header built from `crawler.contact_email`.

## Consequences

- Positive: the product is defensible to legal review; nothing licensed is
  ingested, and analyst content is visibly third-party commentary.
- Positive: SSRF, robots and rate policy are enforced in one place for every
  source, including redirects.
- Negative: briefs contain less analyst context than a human with a research
  licence could add. Organisations holding a licence should integrate through
  the vendor's sanctioned API or data feed as a new `research/sources` module
  rather than widening `DEFAULT_ANALYST_POLICIES`.
- Negative: sites that block the crawler in robots.txt produce no evidence; the
  verdict then trends to `NOT_ENOUGH_EVIDENCE`, which is the correct signal.
- Operational: `crawler.contact_email` defaults to a placeholder
  (`research-agent@example.org`). Staging and prod settings reject it at load
  time; jobs receive the real address from the bundle variable
  `sec_contact_email` (`--contact-email`) and the serving endpoint from the secret
  `client-research-agent/sec-contact-email` (`CRA_CRAWLER__CONTACT_EMAIL`).
