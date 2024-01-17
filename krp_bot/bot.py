import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

from apscheduler.schedulers.blocking import BlockingScheduler

import telegram
telegram_token = ''
telegram_chat_id = ''
bot = telegram.Bot(token=telegram_token)

options = Options()

options.add_argument("--headless")
options.add_argument("--window-size=1920,1080")
options.add_argument('log-level=3')

c = DesiredCapabilities.CHROME
c['pageLoadStrategy'] = 'none'

driver = webdriver.Chrome(options=options, executable_path='', desired_capabilities=c)
w  = WebDriverWait(driver, 6000)

actionChains = ActionChains(driver)

def refresh():
    driver.refresh()

def login():
    # 로그인 하던 부분은 처음에 한번만 실행되게 밖으로 빼기
    driver.get('https://krp.kaist.ac.kr')
    w.until(EC.presence_of_element_located((By.XPATH, '//*[@id="wrap_main"]/div[1]/div/a')))
    driver.find_element(By.XPATH, '//*[@id="wrap_main"]/div[1]/div/a').click()

    w.until(EC.presence_of_element_located((By.XPATH, '//*[@id="IdInput"]')))
    driver.find_elements_by_id('IdInput')[0].click()
    driver.find_elements_by_id('IdInput')[0].send_keys('아이디')
    driver.find_element(By.XPATH, '/html/body/div/div/div[2]/div/div/fieldset/ul/li[2]/input[1]').click()

def main(do_refresh=False):
    if do_refresh:
        refresh()
    # 아래의 코드 구문을 통해서 우리가 원하는 정보가 담긴 box 가 로딩될때까지 대기한다.
    w.until(EC.presence_of_element_located((By.XPATH, '//*[@id="container"]/div/div[3]')))

    boxes = driver.find_element(By.XPATH, '//*[@id="container"]/div/div[3]').find_elements_by_class_name('box')
    time.sleep(1)
    information = {}
    for box in boxes:
        category = box.text.split('\n')[0].replace(' ', '')
        content = box.find_elements_by_class_name('txt_box')[0].text
        if content == '':
            try:
                content = ' '.join(box.text.split('\n')[1:])
            except:
                pass
        information[category] = content
        if len(box.find_elements_by_class_name('mini_txt_box')) > 0:
            mini_text = box.find_element_by_class_name('mini_txt_box').text
            category = mini_text.split(':')[0].replace(' ', '')
            content = list(map(int, mini_text.split(':')[1:]))
            information[category] = content
    print(information)

    elapse = information['오늘예상누적']
    mystate = [elapse[0], elapse[1]]
    information['오늘예상누적'] = f'{elapse[0]}H {elapse[1]}M'
    remain = information['금주잔여복무시간(목표:40H)'].split('H')
    if remain[1] == '':
        remain[1] = '00'
    remtime = max(int(remain[0])*60 + int(remain[1].replace('M','').replace(' ','')) - mystate[0]*60 - mystate[1], 0)
    information['남은시간'] = f'{remtime//60}H {remtime%60}M'

    final_text = ''
    for key, info in information.items():
        final_text += f'{key} : {info}\n'

    from datetime import datetime
    now = datetime.now()

    if information['퇴근시간'] != '':
        print('퇴근하였습니다.')
    elif remtime != 0 and now.hour >= 10 and information['출근시간'] == '':
        bot.sendMessage(chat_id=telegram_chat_id, text='오늘은 아직 출근하지 않았습니다!!!')    
    else:
        bot.sendMessage(chat_id=telegram_chat_id, text=final_text)

    return 0
    

    
if __name__ == '__main__':
    login()
    main()
    from functools import partial
    from apscheduler.schedulers.blocking import BlockingScheduler
    scheduler = BlockingScheduler()
    main_job = partial(main, True)
    scheduler.add_job(main_job, 'cron', minute='*/1')
    scheduler.add_job(refresh, 'cron', minute='1-29,31-59/1')
    scheduler.start()
    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            driver.quit()
            scheduler.shutdown()
            break
    
