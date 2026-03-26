Get matchs already played with results: UPDATE IN PROGRESS

====== INTRODUCTION ======

MAIN Upgrades: 
-search field in scraper.

-More accuracy in predictor.

-Benchmark made decision in prédictor(result in the bengining of lower log)

Flashscore scores scraper/parser, with over and correct scores predictors.

IMPORTANT: you will get an error message if you try to load a csv with not enough scores inside, you will just have to click "ok".

Benchmark in "PREDICTOR(F4).py" => output file is "Benchmark_F4.txt".

In "PREDICTOR(F4).py" you can write results from the GUI to have file for backtesting.

In "PREDICTOR(F4).py": message error export to a texte file(save_error_to_csv) by clicking "ok" button.

You will find:
-14 CSV files with results.
-6 Python scripts: 2 predictors(main predictor(F4) + correct score and 3 scrapers.
-1 sh file
-Readme.

====== SCRAPERS ======

Match selection + leagues by country or continent.

Scraper is set to collect 320 scores maximum by match(confrontation/game) and by team.
It get home and away scores and write it in column A and column C of a ".CSV" file.
Older scores are writen in the begining of the file.

Can be upgraded to more than 320.

The amount of scores collected will collect the same amount for the 2 team => the amount of scores of the team who has less played matchs.

All teams scores ends with 0, because it is the match of the day, no yet played so 0 goal, it's sometime needed to make the predictions computing.

Fully automatic, get all teams scores available on flashscore.

4 sports, football(soccer), basketball, ice hockey, american football.

You can try to add other sports if you want.

3 scraping scripts, 3 options:
1."FLASHSCORE TODAY SCRAPER.py":
Time selection is needed, from 0.00 to 23.59 get all/or in a time range scores for the running day.

2."FLASHSCORE LEAGUE SCRAPER.py":
Time and country is needed, date selection avoid scraping all month matchs get scores from specific(s) league(s) using DAY and TIME.


3."FLASHSCORE MATCH SCRAPER.py": 
get 2 teams scores with a link kind "https://www.flashscore.fr/match/football/espanyol-QFfPdh1J/oviedo-SzYzw34K/?mid=A90Ay4iL".

How to use: 
FIRST: scan flashscore...wait about 20 seconds...=> see the availlable matchs in the log.
To collect scores: click "get scores", wait until complete(about 15 second each 640 scores: 2 teams).


====== PREDICTOR ======

Let's go talking about PREDICTOR => "PREDICTOR(F4).py":
It read CSV and compute the prediction.

I totally build prediction algorithm, chatGPT helped me to code it.


====== MORE INFO ======

CSV export name exemple: 1100_Foot_320_Ostrava_B_VS_Polonia_B.csv = TIME_SPORT_number of scores_ teamA_teamC.csv

I you want to know more about technical information, paste full scripts to chatGPT. 

To add features, try chatGPT.

Contact: galaxiea20maj999@gmail.com

Thank you, bye!
