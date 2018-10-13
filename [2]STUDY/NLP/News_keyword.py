'''동아일보 특정 키워드를 포함하는, 특정 날짜 이전 기사 내용 크롤러(정확도순 검색)
python [모듈이름][키워드][가져올 페이지 숫자][결과 파일명]
한페이지에 기사 15개'''

import sys
from bs4 import BeautifulSoup
import urllib.request
from urllib.parse import quote

'''quote는 urlopen에서 인자로 사용되는 URL주소(이하 타겟 주소)에 한글(UTF_8)이 포함되었을 때, 이를 아스키형식으로 바꿔주기 위한함수'''
TARGET_URL_BEFORE_PAGE_NUM = "http://news.donga.com/search?p="
TARGET_URL_BEFORE_KEYWORD = '&query='
TARGET_URL_REST = '&check_news=1&more=1&sorting=3&search_date=1&v1=&v2=&range=3'

# 기사 검색 페이지에서 기사 제목에 링크된 기사 본문 주소 받아오기
# 원하는 page_num가 10이면 i의 최댓값은 9가 되고, 그때의 'current_page_num'은 136이 된다.
# class가 'tit'인 p태그 안의 첫번째 a채그에 연결된 URL주소가 해당 기사 본문 URL가 포함된 것을 알수 있다.
# 따라서 우리는 위에서 만든 'soup'객체에서 class=tit인 p태그를 모두 뽑아와 그 안에 있는 첫번째 a태그의 'href'의 내용을 가져온다면 모든 기사의 내용을 크롤링할 수 있따.
def get_link_from_news_title(page_num, URL, output_file):
    for i in range(page_num):
        current_page_num = 1 + i*15
        position = URL.index('=')
        URL_with_page_num = URL[: position+1] + str(current_page_num) + URL[position+1 :]
        source_code_from_URL = urllib.request.urlopen(URL_with_page_num)
        soup = BeautifulSoup(source_code_from_URL, 'lxml', from_encoding='utf-8')
        for title in soup.find_all('p','tit'):
            title_link = title.select('a')
            article_URL = title_link[0]['href']
            get_text(article_URL, output_file)

# 기사 본문 내용 긁어오기 (위 함수 내부에서 기사 본문 주소 받아 사용되는 함수)
def get_text(URL, output_file):
    source_code_from_url = urllib.request.urlopen(URL)
    soup = BeautifulSoup(source_code_from_url, 'lxml', from_encoding='utf-8')
    content_of_article = soup.select('div.article_txt')
    for item in content_of_article:
        string_item = str(item.find_all(text=True))
        output_file.write(string_item)


# 메인함수
def main(argv):
    if len(argv) != 4:
        print("python [모듈이름] [키워드] [가져올 페이지 숫자] [결과 파일명]")
        return
    keyword = argv[1]
    page_num = int(argv[2])
    output_file_name = argv[3]
    target_URL = TARGET_URL_BEFORE_PAGE_NUM + TARGET_URL_BEFORE_KEYWORD \
                 + quote(keyword) + TARGET_URL_REST
    output_file = open(output_file_name, 'w')
    get_link_from_news_title(page_num, target_URL, output_file)
    output_file.close()


if __name__ == '__main__':
    main(sys.argv)
