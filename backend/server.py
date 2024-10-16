import time
from flask import Flask, jsonify
from flask_caching import Cache
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import webdriver_manager
from webdriver_manager.chrome import ChromeDriverManager


app = Flask(__name__)

cache = Cache(app, config={
    'CACHE_TYPE': 'SimpleCache',
    'CACHE_DEFAULT_TIMEOUT': 300
})

@app.route('/scrape')
@cache.cached(timeout=300)
def scrape():
    options = webdriver_manager.ChromeOptions()
    options.add_argument('--headless')
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    
    try:
        driver.get('https://gsm24.pl/')
        time.sleep(5)
        data = driver.page_source
    finally:
        driver.quit()
        
    return jsonify(data)

if  __name__ == '__main__':
    app.run(host='0.0.0.0',port=5002,debug=True)