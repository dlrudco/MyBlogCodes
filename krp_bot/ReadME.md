# KRP Bot
한국과학기술원, KAIST의 전문연구요원 복무관리를 위한 텔레그램 봇의 코드입니다. 타 기관의 전문연구요원은 참고삼아 볼 수 있지만, 도움이 되지 않을 수 있습니다 :)

본 깃 레포지토리가 도움이 되었다면 스타★ 부탁드립니다 :)

오라클 무료 클라우드 인스턴스를 사용하시려는 분은 블로그[링크](https://doodlrudco.tistory.com/45) 를 참조해주시기 바랍니다.

## Required Packages(윈도우의 경우 screen을 제외하고 개별 설치)
일반 설치 패키지 : git, screen, google-chrome-stable

설치파일 기반 패키지 : miniconda(오라클 클라우드를 사용하는경우 권장), or anaconda

python package : selenium, python-telegram-bot

다운로드 해야할 것 : chromedriver(설치된 크롬 브라우저와 버전을 맞출 것)

## How to Use
 1. 오라클 클라우드 인스턴스를 생성합니다. 혹은, 사용가능한 개인용 PC 또는 서버를 준비합니다.
 2. 아나콘다 혹은 미니콘다를 설치합니다.(Optional, 호스트 파이썬 환경을 사용해도 무방. 권장사항)
 3. 일반 설치 패키지들을 설치해줍니다 :
    ```shell
    sudo apt-get update && sudo apt-get install git screen google-chrome-stable
    ```
 5. 본 레포지토리를 다운로드 합니다 :
    ```shell
    git clone https://github.com/dlrudco/MyBlogCodes.git
    ```
 6. krp_bot의 폴더로 이동합니다 :
    ```shell
    cd MyBlogCodes/krp_bot
    ```
 7. 텍스트 에디터를 이용하여 코드를 수정합니다(telegram token과 chat id, 그리고 chromedriver executable path를 수정해줍니다). :
    ```shell
    vim bot.py
    ```
 8. 새로운 스크린 프로세스를 실행합니다. : 
    ```shell
     screen -R krp_bot
    ```
 9. 아나콘다를 설치한 상태이거나 가상환경을 사용한다면 사용할 가상환경을 활성화 합니다 : 
    ```shell
    conda activate krp_bot_env
    python -m pip install python-telegram-bot selenium
    ```
 10. bot.py 파일을 실행합니다 :
     ```shell
     python bot.py
     ```
 12. screen 에서 빠져나옵니다 : Ctrl + a + d
 13. 모바일 앱을 통해서 인증을 수행하면 이후 지속적으로 복무 모니터링이 실행됩니다.
