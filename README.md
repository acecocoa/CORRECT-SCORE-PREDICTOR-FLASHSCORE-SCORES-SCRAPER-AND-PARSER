IMPORTANT: you need to scan before getting scores!

Don't worry if you select 320 scores and you get less, the scraper gets the same amount of scores for the 2 teams, for exemple if you select 320 scores to get and in the confrontation a team have played less and the other have played at least 320, you will get the number of score of the team who played less than 320 for the 2 teams.

Of course you can change it with chatGPT.

CSV export name exemple: 1100_Foot_320_Ostrava_B_VS_Polonia_B.csv = TIME_SPORT_number of scores_ teamA_teamC.csv

Scraper is set on 320 maximum scores by match(confrontation/game) and by team.

Can be upgraded to more than 320.

You can notice than all team scores ends with 0, because it is the match of the day, no yet played so 0 goal, it's sometime needed to make the predictions computing.

Fully automatic, get all teams scores available on flashscore.

4 sports, football(soccer), basketball, ice hockey, american football.

3 scripts, 3 options:
1."FLASHSCORE TODAY SCRAPER.py":
Time selection is needed, from 0.00 to 23.59 get all/or in a time range scores for the running day.

2."FLASHSCORE LEAGUE SCRAPER.py":
Time and country is needed, date selection avoid scraping all month matchs get scores from specific(s) league(s) using DAY and TIME.


3."FLASHSCORE MATCH SCRAPER.py": 
get 2 teams scores with a link kind "https://www.flashscore.fr/match/football/espanyol-QFfPdh1J/oviedo-SzYzw34K/?mid=A90Ay4iL".

How to use: 
Scan flashscore, see in the log.
Click "get scores", wait until complete.

HOW SCRAPERS WORKS:
SCRAPERS works with playwright.

It get home and away scores and write it in column A and column C of a ".CSV" file.
Older scores are writen in the begining of the file.

Let's go talking about PREDICTOR => "PREDICTOR(F4).py":
It read CSV and compute the prediction.

I totally build prediction algorithm, chatGPT helped me to code it.

I you want to know more, paste full scripts to chatGPT. 





