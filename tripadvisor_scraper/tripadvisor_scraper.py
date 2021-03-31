from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException

from bs4 import BeautifulSoup
import pandas as pd
import time


# ----------------------------------------------------------------------------


# Base URL
BASE_URL = 'https://www.tripadvisor.co.uk/Hotel_Review-'


# Destinations
LONDON = '-London_England.html'
PARIS = '-Paris_Ile_de_France.html'
BERLIN = '-Berlin.html'


# All hotel urls
urls = [
    # London
    f'{BASE_URL}g186338-d193072-Reviews-Ambassadors_Hotel{LONDON}',
    f'{BASE_URL}g186338-d248811-Reviews-Travelodge_London_Docklands{LONDON}',
    f'{BASE_URL}g186338-d286746-Reviews-Premier_Inn_London_Kensington_Earl_s_Court_Hotel{LONDON}',
    f'{BASE_URL}g186338-d1110692-Reviews-Viking_Hotel{LONDON}',
    f'{BASE_URL}g186338-d11833714-Reviews-Good_Hotel_London{LONDON}',
    f'{BASE_URL}g186338-d193150-Reviews-OYO_Huttons_Hotel{LONDON}',
    f'{BASE_URL}g186338-d280557-Reviews-1_Lexham_Gardens{LONDON}',
    f'{BASE_URL}g186338-d195394-Reviews-Britannia_Hampstead_Hotel{LONDON}',
    f'{BASE_URL}g186338-d3296647-Reviews-Ibis_Budget_London_Whitechapel_Brick_Lane{LONDON}',
    f'{BASE_URL}g186338-d210757-Reviews-Best_Western_Chiswick_Palace_Suites{LONDON}',
    f'{BASE_URL}g186338-d192104-Reviews-Cromwell_International_Hotel{LONDON}',
    
    # Paris
    f'{BASE_URL}g187147-d197627-Reviews-Ibis_Paris_Alesia_Montparnasse_14eme{PARIS}',
    f'{BASE_URL}g187147-d264890-Reviews-Hotel_Aida_Marais{PARIS}',
    f'{BASE_URL}g187147-d197684-Reviews-Ibis_Paris_Porte_De_Montreuil{PARIS}',
    f'{BASE_URL}g187147-d197486-Reviews-Novotel_Paris_Gare_de_Lyon{PARIS}',
    f'{BASE_URL}g187147-d197495-Reviews-Ibis_Paris_Gare_de_Lyon_Ledru_Rollin_12eme{PARIS}',
    f'{BASE_URL}g187147-d271839-Reviews-Hotel_Muguet{PARIS}',
    f'{BASE_URL}g187147-d197985-Reviews-Pullman_Paris_Eiffel_Tower_Hotel{PARIS}',
    f'{BASE_URL}g187147-d664659-Reviews-Hipotel_Paris_Belleville_Gare_de_l_Est{PARIS}',
    f'{BASE_URL}g187147-d282315-Reviews-Est_Hotel_Paris{PARIS}',
    f'{BASE_URL}g187147-d280081-Reviews-Hotel_Altona{PARIS}',
    f'{BASE_URL}g187147-d242958-Reviews-New_Hotel{PARIS}',
    
    # Berlin
    f'{BASE_URL}g187323-d543299-Reviews-Hotel_Berlin_Central_District{BERLIN}',
    f'{BASE_URL}g187323-d1230216-Reviews-Select_Hotel_Berlin_Gendarmenmarkt{BERLIN}',
    f'{BASE_URL}g187323-d199422-Reviews-AMERON_Berlin_ABION_Spreebogen_Waterside{BERLIN}',
    f'{BASE_URL}g187323-d1485637-Reviews-Adina_Apartment_Hotel_Berlin_Mitte{BERLIN}',
    f'{BASE_URL}g187323-d1963628-Reviews-SANA_Berlin_Hotel{BERLIN}',
    f'{BASE_URL}g187323-d483216-Reviews-A_o_Berlin_Mitte{BERLIN}',
    f'{BASE_URL}g187323-d277122-Reviews-Generator_Berlin_Prenzlauer_Berg{BERLIN}',
    f'{BASE_URL}g187323-d200345-Reviews-AZIMUT_Hotel_Kurfuerstendamm_Berlin{BERLIN}',
    f'{BASE_URL}g187323-d1726547-Reviews-Upper_Room_Hotel{BERLIN}',
    f'{BASE_URL}g187323-d1028471-Reviews-Klassik_Hotel{BERLIN}',
    f'{BASE_URL}g187323-d228285-Reviews-Quentin_Berlin_Hotel{BERLIN}'
]


# Create empty list for reviews and ratings
review_list = []


# Set options for webdriver
options = Options()
options.headless = True


# Start scraping process
try:
    
    # Iterate through urls
    for i, url in enumerate(urls, 1):
        
        # Start timer
        start = time.time()

        # Instantiate driver
        driver = webdriver.Firefox(options=options)

        # Create explicit wait
        wait = WebDriverWait(driver, 10)
        
        # Get website
        driver.get(url)

        # Get total number of pages
        total_pages = int(driver.find_elements_by_class_name("pageNum")[-1].text)

        # Loop through pages
        for page in range(total_pages):

            # Click ok on cookie banner             
            if driver.find_elements_by_id("_evidon-accept-button"):
                wait.until(EC.element_to_be_clickable((By.ID, "_evidon-accept-button"))).click()

            # Click all 'read more' buttons (try 3 times if element is stale)
            error = False
            for j in range(4):
                try:
                    read_more = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "_3maEfNCR")))
                    read_more.click()
                    break
                except (StaleElementReferenceException, TimeoutException):
                    if j < 2:
                        print(f'Element Stale... Trying Again...({j+1}/3)')
                    elif j == 2:
                        print(f'Element Stale... Trying Again...({j+1}/3)')
                        driver.refresh()
                    else:
                        error = True
                        break
            
            # Move to next page if unable to scrape reviews
            if error:
                print('Unable to scrape reviews on page {page} of URL {i}... Moving to next page.')
                continue

            # Get page source
            source = driver.page_source

            # Create soup object
            soup = BeautifulSoup(source)

            # Parse reviews
            web_reviews = soup.find_all('q', class_='IRsGHoPm')
            reviews = [review.text for review in web_reviews]

            # Parse ratings
            web_ratings = soup.find_all('div', attrs={'data-test-target':'review-rating'})
            ratings = [rating.span.attrs['class'][1].split('_')[1][0] for rating in web_ratings]

            # Add review and ratings to list
            review_list.extend([review for review in list(zip(reviews,ratings))])

            # Check if last page
            if driver.find_elements_by_class_name("ui_button.nav.next.primary.disabled"):
                break
                
            # Go to next page if not last
            wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "ui_button.nav.next.primary"))).click()
            
        # End timer & calculate elapsed time
        end = time.time()
        mins, secs = divmod(end-start, 60)
        elapsed = '%d:%02d' % (mins, secs)

        # Notify when url is scraped
        print(f'{str(i)} URL(s) Scraped - Elapsed Time for URL ({i}) --> ({elapsed})')
        
        # Close page to free up memory
        driver.close()

# Close page and quit
finally:
    driver.quit()
    

# Store reviews and ratings in dataframe
df = pd.DataFrame(review_list, columns=['Review', 'Rating'])


# Drop any duplicates if they exist
df = df.drop_duplicates()


# Export dataframe as csv
df.to_csv('../data/scraped_tripadvisor_hotel_reviews.csv', index=False)
