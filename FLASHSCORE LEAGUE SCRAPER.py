#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio
import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
from playwright.async_api import async_playwright
import os
import pandas as pd
from collections import Counter
import re
import unicodedata
import json
from tkinter import simpledialog

FLASHSCORE_RESULTS_MAP = {"fr": "resultats", "com": "results"}
FLASHSCORE_TEAM_MAP = {"fr": "equipe", "com": "team"}

FLASHSCORE_SPORTS = {
    "1 - Football": "https://www.flashscore.fr/",
    "2 - Basketball": "https://www.flashscore.fr/basket/",
    "3 - Hockey/glace": "https://www.flashscore.fr/hockey/",
    "4 - Football Américain": "https://www.flashscore.fr/football-americain/"
}

CSV_SPORT_PREFIX = {
    "1 - Football": "Foot",
    "2 - Basketball": "Basket",
    "3 - Hockey/glace": "Hockey/g",
    "4 - Football Américain": "Foot Am"
}

FLASHSCORE_COUNTRIES = [
    "afrique-du-sud","albanie","algerie","allemagne","andorre","angleterre","angola",
    "antigua-barbuda","arabie-saoudite","argentine","armenie","aruba","australie",
    "autriche","azerbaidjan","bahrein","bangladesh","barbade","belgique","benin",
    "bermudes","bhoutan","bielorussie","bolivie","bosnie-herzegovine","botswana",
    "bresil","bulgarie","burkina-faso","burundi","cambodge","cameroun","canada",
    "cap-vert","chili","chine","chypre","colombie","congo","coree-du-sud",
    "costa-rica","cote-d-ivoire","croatie","danemark","ecosse","egypte",
    "emirats-arabes-unis","equateur","espagne","estonie","eswatini","ethiopie",
    "fidji","finlande","france","gabon","gambie","georgie","ghana","gibraltar",
    "grece","grenade","guatemala","guinee","honduras","hongrie","inde",
    "indonesie","iran","irak","irlande","islande","israel","italie","jamaique",
    "japon","jordanie","kazakhstan","kenya","kosovo","koweit","kyrgyzstan",
    "laos","lesotho","lettonie","liban","liberia","libye","liechtenstein",
    "lituanie","luxembourg","macao","macedoine-du-nord","malawi","malaisie",
    "mali","malte","maroc","martinique","maurice","mauritanie","mexique",
    "moldavie","mongolie","montenegro","mozambique","myanmar","nepal",
    "nicaragua","niger","nigeria","norvege","nouvelle-zelande","oman",
    "ouganda","ouzbekistan","pakistan","palestine","panama","paraguay",
    "pays-bas","pays-de-galles","perou","philippines","pologne","portugal",
    "qatar","rd-congo","republique-dominicaine","republique-tcheque",
    "roumanie","russie","rwanda","salvador","san-marin","senegal","serbie",
    "seychelles","sierra-leone","singapour","slovaquie","slovenie","somalie",
    "sri-lanka","soudan","suede","suisse","suriname","syrie","tadjikistan",
    "taiwan","tanzanie","tchad","thailande","togo","trinite-et-tobago",
    "tunisie","turkmenistan","turquie","ukraine","uruguay","usa",
    "venezuela","viet-nam","yemen","zambie","zimbabwe"
]

def build_flashscore_url(sport_label: str, country: str) -> str:
    sport_map = {
        "1 - Football": "football",
        "2 - Basketball": "basket",
        "3 - Hockey/glace": "hockey",
        "4 - Football Américain": "football-americain",
    }

    sport_part = sport_map.get(sport_label, "")
    base_url = "https://www.flashscore.fr/"

    if sport_part:
        base_url += sport_part + "/"
    if country: 
        base_url += country + "/"

    return base_url

def ensure_results_url(url: str) -> str:
    if not url:
        return url
    url = url.strip()
    if re.search(r"/(" + "|".join(FLASHSCORE_RESULTS_MAP.values()) + r")/?$", url):
        return url
    m = re.search(r"flashscore\.([a-z]+)", url.lower())
    tld = m.group(1) if m else "com"
    results_word = FLASHSCORE_RESULTS_MAP.get(tld, "results")
    if not url.endswith("/"):
        url += "/"
    return url + results_word + "/"

def generate_team_urls_from_duel(duel_url):
    if not duel_url:
        raise ValueError("URL de duel manquante")

    try:
        base_url = duel_url.split("?")[0].rstrip("/")
        segments = base_url.split("/")

        if len(segments) < 7:
            raise ValueError("URL confrontation trop courte")

        def extract_slug_id(segment):
            idx = segment.rfind("-")
            if idx == -1:
                raise ValueError(f"Impossible de trouver l'ID dans {segment}")
            return segment[:idx], segment[idx+1:]

        team_a_slug, team_a_id = extract_slug_id(segments[5])
        team_c_slug, team_c_id = extract_slug_id(segments[6])

        tld = "fr"
        team_word = "equipe"
        results_word = "resultats"

        base_url_team = f"https://www.flashscore.{tld}/{team_word}/"
        url_a = f"{base_url_team}{team_a_slug}/{team_a_id}/{results_word}/"
        url_c = f"{base_url_team}{team_c_slug}/{team_c_id}/{results_word}/"

        return url_a, url_c

    except Exception as e:
        raise ValueError(f"Impossible de générer les URLs des équipes : {e}")

def normalize_name(name):
    name = re.sub(r"\s*\([^)]*\)", "", name)
    name = name.lower().strip()
    return ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')

def time_to_minutes(t):
    if not t or ":" not in t:
        return None
    try:
        h, m = map(int, t.split(":"))
        if 0 <= h < 24 and 0 <= m < 60:
            return h*60 + m
    except:
        return None
    return None

def parse_score_float(s):
    if not s:
        return 0.0
    s = s.strip().replace(",", ".")
    try:
        return float(s)
    except:
        return 0.0

def detect_team_name(results):
    if not results:
        return None
    names = []
    for m in results[:4]:
        names.append(m["home"])
        names.append(m["away"])
    return Counter(names).most_common(1)[0][0]

async def fetch_scores_oneshot(page, url, max_matches, gui_log=None):
    await page.route(
        "**/*",
        lambda route: route.abort()
        if route.request.resource_type in ("image", "font", "media")
        else route.continue_()
    )
    await page.goto(url, wait_until="domcontentloaded")
    try:

        if page.is_closed():
            if gui_log:
                gui_log("[WARN] Page fermée – abandon page.evaluate")
            return []

        data = await page.evaluate("""
        async (maxMatches) => {
            function sleep(ms){return new Promise(r=>setTimeout(r,ms));}

            for(let i=0;i<50;i++){
                if(document.querySelectorAll('div[class*="event__match"]').length>0) break;
                await sleep(100);
            }

            let matches = [...document.querySelectorAll('div[class*="event__match"]')];

            while(matches.length < maxMatches){
                if(matches.length >= maxMatches) break;

                const btn = document.querySelector("span.wcl-scores-caption-05_Z8Ux-");
                if(!btn) break;

                btn.click();

                let stableRounds = 0;
                let currentCount = matches.length;
                while(stableRounds < 5){
                    await new Promise(r => setTimeout(r, 200));
                    matches = [...document.querySelectorAll('div[class*="event__match"]')];
                    if(matches.length === currentCount) stableRounds++;
                    else { currentCount = matches.length; stableRounds = 0; }
                }
            }

            matches = matches.slice(0, maxMatches);

            return matches.map(m => ({
                home: m.querySelector("div.wcl-participant_bctDY.event__homeParticipant span")?.textContent?.trim()||"",
                away: m.querySelector("div.wcl-participant_bctDY.event__awayParticipant span")?.textContent?.trim()||"",
                score_home: parseFloat(m.querySelector("span.event__score.event__score--home")?.textContent.replace(",", "."))||0,
                score_away: parseFloat(m.querySelector("span.event__score.event__score--away")?.textContent.replace(",", "."))||0
            }));

        }
        """, max_matches)

        if not data:
            await page.wait_for_selector('div[class*="event__match"]', timeout=5000)
            matches = await page.query_selector_all('div[class*="event__match"]')
            results = []
            for m in matches[:max_matches]:
                home = await m.query_selector_eval("div.wcl-participant_bctDY.event__homeParticipant span","el=>el.textContent")
                away = await m.query_selector_eval("div.wcl-participant_bctDY.event__awayParticipant span","el=>el.textContent")
                sh = await m.query_selector_eval("span.event__score.event__score--home","el=>el.textContent")
                sa = await m.query_selector_eval("span.event__score.event__score--away","el=>el.textContent")
                results.append({
                    "home": home.strip(),
                    "away": away.strip(),
                    "score_home": parse_score_float(sh),
                    "score_away": parse_score_float(sa)
                })
            return results

        return data

    except Exception as e:
        if gui_log: gui_log(f"[ERROR JS] {e}")
        return []

MATCH_SELECTORS = [
    'div[class*="event__match"]',
    'div[class*="match-row"]',       
    'section.match-container div'     
]

TIME_SELECTORS = [
    'div.event__time',
    'span.event__time',
    'div.match-time'                 
]

async def click_prevus(page, gui_log=None):
    try:
        tabs = await page.query_selector_all("div.filters__tab")

        for tab in tabs:
            text = (await tab.text_content() or "").strip().lower()
            #if gui_log:
                #gui_log(f"[TAB] {text}")

            if "prévus" in text or "à venir" in text:
                await tab.scroll_into_view_if_needed()
                await tab.click()
                await page.wait_for_timeout(700)
                if gui_log:
                    gui_log("[INFO] Onglet 'Prévus' cliqué")
                return True

        if gui_log:
            gui_log("[INFO] Onglet 'Prévus' absent sur cette page (normal)")
        return False

    except Exception as e:
        if gui_log:
            gui_log(f"[ERROR] click_prevus : {e}")
        return False

async def expand_and_fetch_matches(page, time_start, time_end, gui_log=None, league_day=None):
    if page.is_closed():
        if gui_log:
            gui_log("[WARN] Page fermée – abandon expand_and_fetch_matches")
        return []

    MATCH_SELECTORS = [
        'div[class*="event__match"]',
        'div[class*="match-row"]',
        'section.match-container div'
    ]

    TIME_SELECTORS = [
        'div.event__time',
        'span.event__time',
        'div.match-time'
    ]

    try:
        clicked_total = 0
        stable_rounds = 0
        while stable_rounds < 2: 
            buttons_clicked = await page.evaluate("""
                () => {
                    let clicked = 0;
                    const buttons = [...document.querySelectorAll("button")];
                    for(const btn of buttons){
                        try {
                            if((btn.textContent||"").toLowerCase().includes("afficher matchs")){
                                btn.scrollIntoView({behavior:'auto', block:'center'});
                                btn.click();
                                clicked++;
                            }
                        } catch {}
                    }
                    return clicked;
                }
            """)
            clicked_total += buttons_clicked
            #if gui_log:
                #gui_log(f"🔽 Accordéons 'Afficher matchs' cliqués : {buttons_clicked}")
            if buttons_clicked == 0:
                stable_rounds += 1
            else:
                stable_rounds = 0
            await page.wait_for_timeout(1000)  

        if gui_log:
            gui_log(f"[INFO] Total accordéons cliqués : {clicked_total}")
            gui_log("📜 Scan terminé")

        await page.evaluate("""
            async () => {
                const sleep = ms => new Promise(r => setTimeout(r, ms));
                for(let i=0;i<30;i++){
                    window.scrollBy(0, window.innerHeight);
                    await sleep(400);
                }
            }
        """)

        await page.wait_for_selector('div[class*="event__match"]', timeout=5000)

        matches = await page.evaluate(f"""
        () => {{
            const out = [];
            const matchSelectors = {MATCH_SELECTORS};
            const timeSelectors = {TIME_SELECTORS};
            const selectedDay = "{league_day}";
            let matchEls = [];

            for(const sel of matchSelectors){{
                matchEls = [...document.querySelectorAll(sel)];
                if(matchEls.length) break;
            }}

            for(const m of matchEls){{
                const linkEl = m.querySelector("a[href*='/match/']") || m.querySelector("a.eventRowLink");
                if(!linkEl || !linkEl.href) continue;

                let timeEl = null;
                for(const tsel of timeSelectors){{
                    timeEl = m.querySelector(tsel);
                    if(timeEl) break;
                }}
                if(!timeEl) continue;

                const timeText = timeEl.textContent.trim();
                let day = null, hour = null;

                let m1 = timeText.match(/^(\d{{2}}):(\d{{2}})$/);
                let m2 = timeText.match(/^(\d{{2}})\.(\d{{2}})\.\s*(\d{{2}}):(\d{{2}})$/);

                if(m1){{ hour = m1[1]+":"+m1[2]; }}
                else if(m2){{ day = m2[1]; hour = m2[3]+":"+m2[4]; }}
                else {{ continue; }}

                if (selectedDay) {{
                    if (!day) continue;
                    if (day !== selectedDay) continue;
                }}

                const leagueEl = m.closest("section.event")?.querySelector("div.event__title");
                const league = leagueEl ? leagueEl.textContent.trim() : "";

                out.push({{url: linkEl.href, time: hour, day: day, league: league}});
            }}
            return out;
        }}
        """)

        filtered = []
        for m in matches:
            try:
                h, mn = map(int, m["time"].split(":"))
                minutes = h*60 + mn
            except:
                continue
            if time_start <= minutes <= time_end:
                filtered.append(m)

        filtered.sort(key=lambda x: x["time"])
        if gui_log:
            gui_log(f"[INFO] {len(filtered)} matchs trouvés après expansion")
        return filtered

    except Exception as e:
        if gui_log:
            gui_log(f"[ERROR] expand_and_fetch_matches : {e}")
        return []

def build_csv_duel(team_a, scores_a, team_c, scores_c,
                   export_dir, match_time=None,
                   prefix=None, subfolder=None, gui_log=None):

    scores_a = scores_a[::-1]
    scores_c = scores_c[::-1]
    min_len = min(len(scores_a), len(scores_c))
    scores_a = scores_a[:min_len]
    scores_c = scores_c[:min_len]

    NUM_ROWS = 321
    rows = [["", "", ""] for _ in range(NUM_ROWS)]
    col_a, col_c = 0, 2
    start_idx = max(NUM_ROWS - min_len, 1)
    rows[start_idx - 1][col_a] = "A"
    rows[start_idx - 1][col_c] = "C"

    for i in range(min_len):
        rows[start_idx + i][col_a] = scores_a[i]
        rows[start_idx + i][col_c] = scores_c[i]

    rows.append([0, "", 0])
    rows.append([team_a, "", team_c])

    target_dir = export_dir
    if subfolder:
        target_dir = os.path.join(export_dir, subfolder)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    safe_a = team_a.replace(" ", "_")
    safe_c = team_c.replace(" ", "_")
    time_prefix = ""
    if match_time:
        time_prefix = match_time.replace(":", "") + "_"

    filename = f"{time_prefix}{prefix}_{safe_a}_VS_{safe_c}.csv"

    full_path = os.path.join(target_dir, filename)

    pd.DataFrame(rows).to_csv(full_path, index=False, header=False, encoding="utf-8-sig")
    if gui_log:
        gui_log(f"[CSV EXPORT] {full_path}")

    return full_path

class FlashscoreGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FLASHSCORE LEAGUE SCRAPER")
        self.geometry("1000x750")
        self.scanned_matches = []




        frame_scores = tk.Frame(self)
        frame_scores.pack(anchor="w", padx=10, pady=(5,0))
        tk.Label(frame_scores, text="SCORES OPTION").grid(row=0, column=0, sticky="w")
        self.combo_mode = ttk.Combobox(frame_scores, values=["1 - TEAM","2 - TOTAL","3 - BOTH"], width=25, state="readonly")
        self.combo_mode.set("1 - TEAM") 
        self.combo_mode.grid(row=0, column=1, padx=5)

        tk.Label(frame_scores, text="SCORES AMOUNT").grid(row=0, column=2, sticky="w")
        self.combo = ttk.Combobox(frame_scores, values=[str(i) for i in range(40,641,40)], width=10, state="readonly")
        self.combo.set("320")
        self.combo.grid(row=0, column=3, padx=5)

        frame_time_day = tk.Frame(self)
        frame_time_day.pack(anchor="w", padx=10, pady=(5, 0))

        tk.Label(frame_time_day, text="TIME(FROM)").grid(row=0, column=0, sticky="w")
        self.time_start = ttk.Combobox(
            frame_time_day,
            values=[""] + [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)],
            width=7,
            state="readonly"
        )
        self.time_start.set("")
        self.time_start.grid(row=0, column=1, padx=5)

        tk.Label(frame_time_day, text="TIME(TO)").grid(row=0, column=2, sticky="w")
        self.time_end = ttk.Combobox(
            frame_time_day,
            values=[""] + [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)] + ["23:59"],
            width=7,
            state="readonly"
        )
        self.time_end.set("")
        self.time_end.grid(row=0, column=3, padx=5)

        tk.Label(frame_time_day, text="DAY").grid(row=0, column=4, sticky="w")
        self.league_day = ttk.Combobox(
            frame_time_day,
            values=[""] + [f"{i:02d}" for i in range(1, 32)], 
            width=5,
            state="readonly"
        )
        self.league_day.set("")
        self.league_day.grid(row=0, column=5, padx=5)

        #tk.Label(frame_time_day, text="<= SELECT TO SORT").grid(row=0, column=6, sticky="w")

        frame_top = tk.Frame(self)
        frame_top.pack(anchor="w", padx=10, pady=(5,0))
        tk.Label(frame_top, text="SPORT").grid(row=0, column=0, sticky="w")
        self.combo_sport = ttk.Combobox(frame_top, values=list(FLASHSCORE_SPORTS.keys()), width=30, state="readonly")
        self.combo_sport.set("1 - Football") 
        self.combo_sport.grid(row=0, column=1, padx=5)

        tk.Label(frame_top, text="COUNTRY").grid(row=0, column=2, sticky="w")
        self.combo_country = ttk.Combobox(
            frame_top,
            values=[""] + FLASHSCORE_COUNTRIES, 
            width=30,
            state="readonly"
        )
        self.combo_country.set("")  
        self.combo_country.grid(row=0, column=3, padx=5)

        #tk.Label(frame_top, text="<= EMPTY=CONTINENT").grid(row=0, column=4, sticky="w")

        frame_buttons = tk.Frame(self)
        frame_buttons.pack(anchor="w", padx=10, pady=5)

        tk.Button(frame_buttons, text="SCAN LEAGUES", command=self.scan_leagues).grid(row=0, column=0, padx=5)

        tk.Button(frame_buttons, text="SHOW MATCH(S)", command=self.show_matches).grid(row=0, column=1, padx=5)

        tk.Button(frame_buttons, text="GET SCORE(S)", command=self.run_scores).grid(row=0, column=2, padx=5)

        tk.Button(frame_buttons, text="DELETE LOG", command=self.reset_urls).grid(row=0, column=3, padx=5)

        self.log = scrolledtext.ScrolledText(self, height=25)
        self.log.pack(fill="both", expand=True, padx=10)

        startup_msgs = [
            "1. Select SPORT and COUNTRY",
            "2. Click SCAN LEAGUES",
            "3. Click SHOW MATCH(S)",
            "4. Select SCORE OPTION (team = actual team scores; total = team + opponent)",
            "5. If needed select TIME and DAY",
            "6. Select SCORE AMOUNT",
            "7. Click GET SCORES",
            "Click DELETE LOG to remove this message"
        ]

        for msg in startup_msgs:
            self.log_msg(msg)

        tk.Label(self, text="LEAGUE(S)").pack(anchor="w", padx=10)
        self.list_leagues = tk.Listbox(self, selectmode=tk.MULTIPLE, height=10)
        self.list_leagues.pack(fill="x", padx=10)

        self.export_dir = "/home/annick/SCRIPTS/export_csv"
        os.makedirs(self.export_dir, exist_ok=True)

    def log_msg(self,msg):
        self.log.insert(tk.END,msg+"\n")
        self.log.see(tk.END)

    def reset_urls(self):
        self.log.delete('1.0', tk.END)
        self.list_leagues.delete(0, tk.END)
        self.scanned_matches.clear()
        self.export_dir = "/home/annick/SCRIPTS/export_csv"
        os.makedirs(self.export_dir, exist_ok=True)

    def auto_fix_url(self,event):
        entry = event.widget
        url = entry.get().strip()
        if not url.startswith("http"): return
        fixed = ensure_results_url(url)
        if fixed != url:
            entry.delete(0,tk.END)
            entry.insert(0,fixed)
            self.log_msg(f"[AUTO] URL normalisée → {fixed}")

    def run_scores(self):

        path = os.path.join(self.export_dir, "leagues.json")
        if not os.path.exists(path):
            self.log_msg("[ERROR] leagues.json introuvable")
            return

        with open(path, "r", encoding="utf-8") as f:
            leagues_dict = json.load(f)

        selected = self.list_leagues.curselection()
        if not selected:
            self.log_msg("[ERROR] Aucune ligue sélectionnée")
            return

        time_start = time_to_minutes(self.time_start.get())
        time_end = time_to_minutes(self.time_end.get())
        league_day = self.league_day.get().strip()
        if not league_day:
            league_day = None

        if time_start is None or time_end is None:
            self.log_msg("[ERROR] Plage horaire invalide")
            return

        nm = int(self.combo.get())
        mode = self.combo_mode.get()
        sport_label = self.combo_sport.get()
        sport_prefix = CSV_SPORT_PREFIX.get(sport_label, "Sport")

        async def runner():
            async with async_playwright() as p:
                browser = await p.firefox.launch(headless=True)
                scanned = []
                page = await browser.new_page()

                for idx in selected:
                    league_name = self.list_leagues.get(idx)
                    url = leagues_dict.get(league_name)

                    if not url:
                        continue

                    if not url.endswith("/"):
                        url += "/"

                    url_calendar = url + "calendrier/"
                    await page.goto(url_calendar, wait_until="domcontentloaded")
                    await page.wait_for_timeout(1200)

                    matches = await page.evaluate("""
                        () => {
                            const out = [];
                            const matchEls = document.querySelectorAll('div[class*="event__match"]');

                            for(const m of matchEls){

                                const link = m.querySelector("a[href*='/match/']");
                                const timeEl = m.querySelector("div.event__time, span.event__time");

                                if(!link || !timeEl) continue;

                                const timeText = timeEl.textContent.trim();

                                let day = null;
                                let hour = null;

                                let m1 = timeText.match(/^(\d{2}):(\d{2})$/);
                                let m2 = timeText.match(/^(\d{2})\.(\d{2})\.\s*(\d{2}):(\d{2})$/);

                                if(m1){
                                    hour = m1[1] + ":" + m1[2];
                                } else if(m2){
                                    day = m2[1];
                                    hour = m2[3] + ":" + m2[4];
                                } else {
                                    continue;
                                }

                                out.push({
                                    url: link.href,
                                    time: hour,
                                    day: day
                                });
                            }

                            return out;
                        }
                    """)

                    for m in matches:

                        duel_url = m["url"]
                        time_match = m["time"]
                        day_match = m.get("day")

                        if league_day:
                            if not day_match:
                                continue
                            if day_match != league_day:
                                continue

                        try:
                            h, mn = map(int, time_match.split(":"))
                            minutes = h * 60 + mn
                        except:
                            continue

                        if not (time_start <= minutes <= time_end):
                            continue

                        try:
                            url_a, url_c = generate_team_urls_from_duel(duel_url)
                        except:
                            url_a, url_c = None, None

                        scanned.append({
                            "time": time_match,
                            "url_duel": duel_url,
                            "url_a": url_a,
                            "url_c": url_c,
                            "league": league_name
                        })

                await page.close()

                if not scanned:
                    await browser.close()
                    return []

                self.log_msg(f"[INFO] {len(scanned)} match(s) ready to get scores.")

                selection = simpledialog.askstring(
                    "Sélect match(s)",
                    "Enter number separated by ';'\n"
                    "EMPTY = ALL"
                )

                if selection is None:
                    self.log_msg("[INFO] Opération annulée")
                    await browser.close()
                    return []

                if selection.strip() != "":
                    try:
                        indexes = [int(x.strip()) for x in selection.split(";") if x.strip().isdigit()]
                    except:
                        self.log_msg("[ERROR] Format invalide. Exemple: 1;5;8")
                        await browser.close()
                        return []

                    selected_scanned = []
                    for i in indexes:
                        if 1 <= i <= len(scanned):
                            selected_scanned.append(scanned[i-1])
                        else:
                            self.log_msg(f"[WARN] Match {i} inexistant → ignoré")

                    scanned = selected_scanned

                if not scanned:
                    self.log_msg("[ERROR] Aucun match sélectionné")
                    await browser.close()
                    return []

                for i, duel in enumerate(scanned, start=1):

                    if not duel["url_a"] or not duel["url_c"]:
                        continue

                    self.log_msg(f"[MATCH {i}/{len(scanned)}] Récupération scores...")

                    page_a = await browser.new_page()
                    page_c = await browser.new_page()

                    try:
                        scores_a, scores_c = await asyncio.gather(
                            fetch_scores_oneshot(page_a, duel["url_a"], nm, gui_log=self.log_msg),
                            fetch_scores_oneshot(page_c, duel["url_c"], nm, gui_log=self.log_msg)
                        )
                    except Exception as e:
                        self.log_msg(f"[ERROR] fetch_scores_oneshot : {e}")
                        scores_a, scores_c = [], []

                    await page_a.close()
                    await page_c.close()

                    if not scores_a or not scores_c:
                        continue

                    team_a = detect_team_name(scores_a) or "EquipeA"
                    team_c = detect_team_name(scores_c) or "EquipeC"

                    team_a_norm = normalize_name(team_a)
                    team_c_norm = normalize_name(team_c)

                    scores_a_team, scores_c_team = [], []
                    scores_a_total, scores_c_total = [], []

                    for m in scores_a:
                        home_norm = normalize_name(m["home"])
                        away_norm = normalize_name(m["away"])
                        scores_a_team.append(
                            m["score_home"] if home_norm == team_a_norm
                            else m["score_away"] if away_norm == team_a_norm
                            else 0
                        )
                        scores_a_total.append(m["score_home"] + m["score_away"])

                    for m in scores_c:
                        home_norm = normalize_name(m["home"])
                        away_norm = normalize_name(m["away"])
                        scores_c_team.append(
                            m["score_home"] if home_norm == team_c_norm
                            else m["score_away"] if away_norm == team_c_norm
                            else 0
                        )
                        scores_c_total.append(m["score_home"] + m["score_away"])

                    min_team = min(len(scores_a_team), len(scores_c_team))
                    min_total = min(len(scores_a_total), len(scores_c_total))

                    if mode.startswith("1"):
                        build_csv_duel(
                            team_a, scores_a_team,
                            team_c, scores_c_team,
                            self.export_dir,
                            match_time=duel["time"],
                            prefix=f"{sport_prefix}_{min_team}",
                            subfolder="Score équipe",
                            gui_log=self.log_msg
                        )

                    elif mode.startswith("2"):
                        build_csv_duel(
                            team_a, scores_a_total,
                            team_c, scores_c_total,
                            self.export_dir,
                            match_time=duel["time"],
                            prefix=f"{sport_prefix}_{min_total}",
                            subfolder="Score total",
                            gui_log=self.log_msg
                        )

                    else:
                        build_csv_duel(
                            team_a, scores_a_team,
                            team_c, scores_c_team,
                            self.export_dir,
                            match_time=duel["time"],
                            prefix=f"{sport_prefix}_{min_team}",
                            subfolder="Score équipe",
                            gui_log=self.log_msg
                        )
                        build_csv_duel(
                            team_a, scores_a_total,
                            team_c, scores_c_total,
                            self.export_dir,
                            match_time=duel["time"],
                            prefix=f"{sport_prefix}_{min_total}",
                            subfolder="Score total",
                            gui_log=self.log_msg
                        )

                await browser.close()
                return scanned

        try:
            self.scanned_matches = asyncio.run(runner())
            self.log_msg(f"[OK] {len(self.scanned_matches)} x2 scores match(s) saved, ({len(self.scanned_matches)}*2) scores series.")
        except Exception as e:
            self.log_msg(f"[ERROR] run_scores : {e}")

    def show_matches(self):

        path = os.path.join(self.export_dir, "leagues.json")
        if not os.path.exists(path):
            self.log_msg("[ERROR] leagues.json introuvable")
            return

        with open(path, "r", encoding="utf-8") as f:
            leagues_dict = json.load(f)

        selected = self.list_leagues.curselection()
        if not selected:
            self.log_msg("[ERROR] Aucune ligue sélectionnée")
            return

        time_start = time_to_minutes(self.time_start.get())
        time_end = time_to_minutes(self.time_end.get())
        league_day = self.league_day.get().strip()

        if not league_day:
            league_day = None

        async def runner():

            async with async_playwright() as p:
                browser = await p.firefox.launch(headless=True)
                page = await browser.new_page()

                matches_all = []

                for idx in selected:

                    league_name = self.list_leagues.get(idx)
                    url = leagues_dict.get(league_name)

                    if not url:
                        continue

                    url_calendar = url.rstrip("/") + "/calendrier/"

                    await page.goto(url_calendar, wait_until="domcontentloaded")
                    await page.wait_for_timeout(1200)

                    matches = await page.evaluate("""
                    () => {

                        const out = [];
                        const matchEls = document.querySelectorAll('div[class*="event__match"]');

                        for(const m of matchEls){

                            const link = m.querySelector("a[href*='/match/']");
                            const timeEl = m.querySelector("div.event__time, span.event__time");

                            if(!link || !timeEl) continue;

                            const timeText = timeEl.textContent.trim();

                            let day=null;
                            let hour=null;

                            let m1=timeText.match(/^(\d{2}):(\d{2})$/);
                            let m2=timeText.match(/^(\d{2})\.(\d{2})\.\s*(\d{2}):(\d{2})$/);

                            if(m1){
                                hour=m1[1]+":"+m1[2];
                            }else if(m2){
                                day=m2[1];
                                hour=m2[3]+":"+m2[4];
                            }

                            out.push({
                                url:link.href,
                                time:hour,
                                day:day
                            });
                        }

                        return out;
                    }
                    """)

                    for m in matches:

                        duel_url = m["url"]
                        time_match = m["time"]

                        try:
                            url_a, url_c = generate_team_urls_from_duel(duel_url)
                        except:
                            url_a, url_c = None, None

                        matches_all.append({
                            "time": time_match,
                            "url_duel": duel_url,
                            "url_a": url_a,
                            "url_c": url_c,
                            "league": league_name
                        })

                await browser.close()
                return matches_all

        self.scanned_matches = asyncio.run(runner())

        if not self.scanned_matches:
            self.log_msg("[INFO] Aucun match trouvé")
            return

        self.log_msg(f"[INFO] {len(self.scanned_matches)} matchs found.")
        self.log_msg(f"[INFO] Now select TIME and DAY")

        for i,m in enumerate(self.scanned_matches,1):

            self.log_msg(f"\n[MATCH {i}] {m['time']} → URL duel: {m['url_duel']}")
            #self.log_msg(f"  URL équipe A : {m['url_a']}")
            #self.log_msg(f"  URL équipe C : {m['url_c']}")

    def scan_leagues(self):

        sport_label = self.combo_sport.get().strip()
        country = self.combo_country.get().strip()

        base_url = build_flashscore_url(sport_label, country)

        async def runner():
            async with async_playwright() as p:

                browser = await p.firefox.launch(headless=True)
                page = await browser.new_page()

                await page.goto(base_url, wait_until="domcontentloaded")
                await page.wait_for_timeout(1000)

                leagues_dict = {}

                headers = await page.query_selector_all(
                    "div.headerLeague__wrapper a.headerLeague__title"
                )

                for h in headers:

                    name = (await h.text_content() or "").strip()
                    if not name:
                        continue

                    href = await h.get_attribute("href")

                    if href:
                        leagues_dict[name] = "https://www.flashscore.fr" + href

                await browser.close()

                return leagues_dict

        try:

            leagues_dict = asyncio.run(runner())

            if not leagues_dict:
                self.log_msg("[INFO] Click 'SCAN LEAGUES' again.")
                return

            # remplir la liste des ligues
            self.list_leagues.delete(0, tk.END)

            for lg in leagues_dict.keys():
                self.list_leagues.insert(tk.END, lg)

            # sauvegarder les ligues
            path = os.path.join(self.export_dir, "leagues.json")

            with open(path, "w", encoding="utf-8") as f:
                json.dump(leagues_dict, f, ensure_ascii=False, indent=2)

            self.log_msg(f"[INFO] {len(leagues_dict)} ligues détectées")

        except Exception as e:
            self.log_msg(f"[ERROR] scan_leagues : {e}")

    def run_async(self, coro):
        asyncio.create_task(coro)

if __name__ == "__main__":
    FlashscoreGUI().mainloop()
