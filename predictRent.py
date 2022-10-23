# -*- coding: utf-8 -*-
"""
크롤링을 위해서 선택한 사이트는 RentHop 사이트 (http://www.renthop.com) 입니다.  

해당 사이트에 가면 올라와있는 매물의 주소, 가격, 침실 수 및 욕실 수를 확인할 수 있습니다. 각 매물에 대해서 이를 크롤링을 할텐데 여기에는 파이썬 requests 패키지를 사용합니다.  

requests의 사용 방법에 대한 개요를 보려면 http://docs.python-requests.org/en/master/user/quickstart/ 에서 가이드를 볼 수 있습니다.
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
# %matplotlib inline

"""뉴욕시 NYC 아파트 데이터를 사용해보겠습니다. 해당 데이터의 URL은 https://www.renthop.com/nyc/apartments-for-rent 입니다. 해당 페이지의 HTML 코드를 가져오는 코드는 다음과 같습니다."""

r = requests.get('https://www.renthop.com/nyc/apartments-for-rent')

"""코드를 가져와서 r객체에 저장 후 content를 통해 가져온 값을 확인할 수 있습니다."""

r.content

from bs4 import BeautifulSoup

"""HTML 코드 분석을 수행하기 위해 BeautifulSoup라는 패키지를 사용합니다."""

soup = BeautifulSoup(r.content, "html5lib")

"""이제 우리가 만든 soup를 사용하여 아파트 데이터를 구문 분석 할 수 있습니다. 가장 먼저 할 일은 페이지의 목록 데이터를 포함하는 div 태그를 검색하는 것입니다. 다음 코드에서 확인할 수 있습니다.  

아래 코드는 div 태그에서 class가 search-info가 들어간 것들을 찾아내서 listing_divs라는 이름의 리스트에 추가합니다.
이것들 각각이 하나의 매물을 포함하고 있는 원소라고 보시면 됩니다.
"""

listing_divs = soup.select('div[class*=search-info]')
listing_divs

len(listing_divs)

"""리스트의 원소는 22개입니다. 또는 다른게 해석하면 해당 페이지에서 총 22개의 매물을 찾았다는 의미입니다.  
그 중 첫번째 원소를 가지고 각 원소로부터 어떻게 데이터를 파싱할지 고민해봅시다.
"""

listing_divs[0]

listing_divs[0].select('a[id*=title]')

listing_divs[0].select('a[id*=title]')[0]['href']

listing_divs[0].select('a[id*=title]')[0].string

listing_divs[0].select('div[id*=hood]')[0]

listing_divs[0].select('div[id*=hood]')[0].string.replace('\n', '')

href = listing_divs[0].select('a[id*=title]')[0]['href']
addy = listing_divs[0].select('a[id*=title]')[0].string
hood = listing_divs[0].select('div[id*=hood]')[0]\
.string.replace('\n','')

print(href)
print(addy)
print(hood)

import re

temp = '''
<div class="font-size-10 b d-inline-block">
$2,850
</div>

기타 등등 여러가지 HTML 코드들
'''

"""문자열이름.index('검색하고자하는 문자열')  
를 사용하시면 검색하고자하는 문자열의 위치가 리턴됩니다.
"""

temp.index('$')

"""위 코드는 temp라는 문자열에서 '$'의 위치. 즉, 인덱스는 어디야? 라고 묻는 코드입니다."""

temp[45:45+100]

temp[45:45+100].split()

temp[45:45+100].split()[0]

"""이제 위 코드를 아래 코드에 적용을 해볼텐데요."""

len(listing_divs[1])

"""listing_divs[0]는 타입이 문자열이 아니라 BeautifulSoup의 객체(함수의 기능을 갖고 있는 변수)입니다.  
BeautifulSoup의 객체에 .text를 하시게 되면 문자열로 바뀌게 됩니다.  

이것은 BeautifulSoup 패키지가 정한 규칙입니다.
"""

type(listing_divs[0].text)

"""타입을 확인하면 문자열로 바뀐 것을 확인할 수 있습니다.

이제 우리가 앞서 사용했던 문자열의 index 기능을 사용할 수 있게 됩니다.
"""

index_num = listing_divs[0].text.index('$') 
contains_word_str = listing_divs[0].text[index_num:index_num+100]

contains_word_list = contains_word_str.split()
contains_word_list[0]

index_num = listing_divs[0].text.index(' Bed\n') 
contains_word_str = listing_divs[0].text[index_num-10:index_num+4]
contains_word_list = contains_word_str.split('\n')
#print(listing_divs[0].text.index(' Bed\n'))
print(contains_word_list[-1])

index_num = listing_divs[0].text.index(' Bath\n') 
contains_word_str = listing_divs[0].text[index_num-10:index_num+5]
contains_word_list = contains_word_str.split('\n')
# print(contains_word_list[0])
print(contains_word_list[-1])

listing_list = []
for idx in range(len(listing_divs)): #한페이지에 매물 수가 22개이므로 listing_divs는 22라는 int형 정수값을 갖는다. 즉 이 for문은 22번 실행된다는 것이다.
    indv_listing = [] #indv_listing 이라는 변수를 리스트으로 포맷 및 초기화
    current_listing = listing_divs[idx] #현재 for문의 진행 순서(카운트값, idx)의 리스트를 current_listing에 할당
    href = current_listing.select('a[id*=title]')[0]['href'] #title이라는 id태그에서 href 정보를 찾아 'href' 변수에 할당
    addy = current_listing.select('a[id*=title]')[0].string #위 코드와 동일 
    hood = current_listing.select('div[id*=hood]')[0]\
    .string.replace('\n','') #위 코드와 동일
    
    #데이터 마이닝을 통해 얻은 데이터들을 미리 만들어둔 indv_listing에 append해준다.
    indv_listing.append(href) 
    indv_listing.append(addy)
    indv_listing.append(hood)

    #아직 완벽하게 가공된 데이터가 아니다. 필요한 데이터만 완전히 추출하기 위해선 다음과 같은 코드를 수행해야 한다.
    # 가격 추가
    try:
      index_num = current_listing.text.index('$') 
      contains_word_str = current_listing.text[index_num:index_num+100]
      contains_word_list = contains_word_str.split()
      indv_listing.append(contains_word_list[0])
    except:
      indv_listing.append('-')

    # 침실 추가
    try:
      index_num = current_listing.text.index(' Bed\n') 
      contains_word_str = current_listing.text[index_num-10:index_num+4]
      contains_word_list = contains_word_str.split('\n')
      indv_listing.append(contains_word_list[-1])
    except:
      indv_listing.append('-')

    # 욕실 추가
    try:
      index_num = current_listing.text.index(' Bath\n') 
      contains_word_str = current_listing.text[index_num-10:index_num+5]
      contains_word_list = contains_word_str.split('\n')
      indv_listing.append(contains_word_list[-1])
    except:
      indv_listing.append('-')

    listing_list.append(indv_listing)

listing_list

len(listing_list)

"""지금까지 실습해본 것은 하나의 페이지에 대해서 수행한 것이고 이제 여러 개의 페이지에 접근하는 방법을 알아봅시다.  
기본적으로 renthop의 주소 구조는 다음과 같습니다.  

https://www.renthop.com/search/nyc?max_price=50000&min_price=0&page=페이지숫자&sort=hopscore&q=&search=0  

여기서 페이지 숫자만 변수로 사용하여서 여러 페이지에 접근하면 되겠지요?
"""

# iterable url

url_prefix = "https://www.renthop.com/search/nyc?max_price=50000&min_price=0&page="
page_no = 1
url_suffix = "&sort=hopscore&q=&search=0"

# test url

for i in range(3):
    target_page = url_prefix + str(page_no) + url_suffix
    print(target_page)
    page_no += 1

def parse_data(listing_divs):
    listing_list = []
    for idx in range(len(listing_divs)):
        indv_listing = []
        current_listing = listing_divs[idx]
        href = current_listing.select('a[id*=title]')[0]['href']
        addy = current_listing.select('a[id*=title]')[0].string
        hood = current_listing.select('div[id*=hood]')[0]\
        .string.replace('\n','')

        indv_listing.append(href)
        indv_listing.append(addy)
        indv_listing.append(hood)

        # 가격 추가
        try:
          index_num = current_listing.text.index('$') 
          contains_word_str = current_listing.text[index_num:index_num+100]
          contains_word_list = contains_word_str.split()
          indv_listing.append(contains_word_list[0])
        except:
          indv_listing.append('-')

        # 침실 추가
        try:
          index_num = current_listing.text.index(' Bed\n') 
          contains_word_str = current_listing.text[index_num-10:index_num+4]
          contains_word_list = contains_word_str.split('\n')
          indv_listing.append(contains_word_list[-1])
        except:
          indv_listing.append('-')

        # 욕실 추가
        try:
          index_num = current_listing.text.index(' Bath\n') 
          contains_word_str = current_listing.text[index_num-10:index_num+5]
          contains_word_list = contains_word_str.split('\n')
          indv_listing.append(contains_word_list[-1])
        except:
          indv_listing.append('-')

        listing_list.append(indv_listing)
    return listing_list

"""위에서 구현한 파싱 함수를 100페이지 동안 호출하도록 합니다."""

all_pages_parsed = []
for i in range(100):
    target_page = url_prefix + str(page_no) + url_suffix
    print(target_page)
    r = requests.get(target_page)
    
    soup = BeautifulSoup(r.content, 'html5lib')
    
    listing_divs = soup.select('div[class*=search-info]')
    
    one_page_parsed = parse_data(listing_divs)
    
    all_pages_parsed.extend(one_page_parsed)
    
    page_no += 1

len(all_pages_parsed)

all_pages_parsed

"""이렇게 쌓인 데이터를 데이터프레임으로 읽어봅시다.

리스트의 원소가 리스트인 이중 리스트는 다음과 같이 pd.DataFrame()으로 감싸주어서 데이터프레임으로 로드가 가능합니다.  
이때, columns라는 인자값으로 컬러명의 리스트를 주면, 각 컬럼에
"""

df = pd.DataFrame(all_pages_parsed, columns=['url', 'address', 'neighborhood', 'rent', 'beds', 'baths'])

df

"""beds열에 있는 데이터의 종류를 출력해봅시다."""

df['beds'].unique()

"""'-'의 경우에는 앞서 Bed라는 단어를 찾아보고 try except문에서 해당 페이지에 Bed를 찾지 못했을 경우에 넣도록 코드를 작성했었습니다. 그 외에는 대체적으로는 잘 들어간 것 같지만 앞에 띄어쓰기가 있는 경우가 있습니다.  

파이썬에서 엄연히 ' 2 Bed'와 '2 Bed'는 다른 문자열입니다. 앞의 문자열은 뒤의 문자열과는 달리 앞에 공백이 존재하기 때문입니다. 마찬가지로 ' 1 Bed'와 '1 Bed'도 다른 단어로 인식되어 출력된 것을 확인할 수 있습니다.  

우선 ' 2 Bed'와 같이 예상하지 못한 값이 들어왔을 때, 이게 정말 제대로 들어간건지가 궁금할 수 있습니다. 그런데 우리는 크롤링 과정에서 URL 열에 해당 페이지의 URL을 수집하였으므로 직접 URL에 들어가서 확인해보면 되겠습니다.  

데이터프레임의 beds열의 값이 ' 2 Bed'인 행의 'url'열의 값을 가져와줘 라는 코드는 아래와 같이 작성할 수 있습니다.
"""

df[df['beds']==' 2 Bed']['url'].values

"""해당 URL에 접속하여 확인한 결과 2 Bed임을 확인했습니다. 그렇다면 어차피 데이터가 1개밖에 되지 않으므로 수작업으로 ' 2 Bed'를 '2 Bed'로 바꿔봅시다. 데이터프레임에서 특정 값에 접근하여 해당 값을 수정하는 방법으로 at이라는 것을 사용할 수 있습니다.  

우선 바꾸고자 하는 값이 있는 위치의 행을 추적합니다.
"""

df[df['beds']==' 2 Bed']

"""그리고 해당 행의 인덱스를 확인합니다. 이 데이터의 인덱스는 369네요.  

데이터프레임의 이름.at['인덱스', '열의 이름']  

을 사용하면 해당 데이터의 값을 가져올 수 있습니다.
"""

df.at[369, 'beds']

"""더 중요한 것은 여기에 다른 값을 덮어쓸 수 있다는 것입니다."""

df.at[369, 'beds'] = '2 Bed'

df.at[369, 'beds']

"""다시 확인해보니 ' 2 Bed'에서 '2 Bed'로 값이 변환된 것을 알 수 있습니다.  
다시 한번 데이터프레임의 beds열에 존재하는 모든 종류의 값들을 출력해봅시다.
"""

df['beds'].unique()

"""이제 더 이상 ' 2 Bed'가 사라진 것을 볼 수 있습니다.  
여러분들이 직접 크롤링 하실 때에도 ' 2 Bed' 외에도 예상하지 못한 값이 크롤링 되어져 있을 수 있습니다.  

그런 경우에는 방금 보신 것과 같이 URL을 꺼내와서 사이트에 들어가보신 후에  
해당 페이지에서 확인한 정상적인 값으로 at을 통해서 바꿔치기 해주시면 되겠습니다.  

일반적으로 잘못 크롤링 된 값은 많지 않으므로 수작업으로도 가능합니다.  

그런데 사실 왼쪽 공백을 제거하는 정도는 값을 굳이 수작업으로 바꿀 필요 없이 lstrip()을 통해서도 가능합니다.  
lstrip()은 문자열에서 앞에 있는 공백을 제거하는 역할을 합니다.  
임의의 문자열에 대해서 왼쪽 공백을 제거하는 실습을 해봅시다.
"""

' 임의의 문자열'.lstrip()

"""beds 열 전체에 대해서 이를 적용해줍니다."""

# 앞의 공백 제거
df['beds'] = df['beds'].apply(lambda x: x.lstrip())

df['beds'].unique()

"""이제 ' 1 Bed'도 사라진 것을 볼 수 있습니다. '1 Bed'로 변경되었기 때문입니다.  
아직 Bed 값이 없었을 때는 '-' 값이 들어가있다는 것만 기억해둡시다.  

마찬가지로 baths열에 대해서도 값의 종류를 확인합니다.
"""

df['baths'].unique()

"""Bed때와 마찬가지로 앞에 공백이 있는 경우가 있습니다. 그 외 특이한 값은 없으므로 굳이 URL 확인 후 at을 통해 수작업으로 값을 수정해줄 필요는 없어보입니다. lstrip()으로 앞의 공백을 제거해줍니다."""

# 앞의 공백 제거
df['baths'] = df['baths'].apply(lambda x: x.lstrip())

df['baths'].unique()

"""결과적으로 전처리를 거친 데이터프레임은 다음과 같습니다."""

df

"""총 2,134개의 행을 가졌으며 6개의 열을 가지고 있습니다. 혹시 중복이 있다면 제거해봅시다."""

df.drop_duplicates(inplace=True)

df

"""열이 1,942개로 줄었습니다. 중복 데이터가 약 200개가 존재했었다는 의미입니다.  
이제 각 열의 데이터 타입을 확인해봅시다.
"""

df.info()

"""모두 문자열 데이터입니다. beds의 열에 '-' 값인 경우에 대해서 데이터프레임을 출력해봅시다."""

df[df['beds']=='-']

"""무려 415개가 beds의 열에 값이 없어서 '-'가 들어가있습니다.  
후에 머신 러닝 모델을 돌려볼 것이기 때문에 rent, beds, baths에 대해서 모두 숫자로 값을 변경해줄 겁니다. 다음과 같은 과정을 거칠 예정입니다.  

* beds의 열에서 ' Bed'를 제거해서 모두 숫자만 남깁니다.  
* beds의 열에서 '-' 값을 가지는 경우 방이 별도로 없다는 의미이므로 숫자 0을 넣습니다.  
* 정수값만 남게된 beds열 전체를 데이터 타입을 정수형(int)로 변환합니다.  
* rent 열에서 %와 ,를 제거하여 숫자만 남도록하고 정수형(int)으로 변환합니다.  
* baths 열에서 ' Bath'를 제거하여 실수형(float)로 변환합니다.  

아래의 코드는 이 과정을 순차적으로 진행하는 코드입니다.
"""

df['beds'] = df['beds'].map(lambda x: x.replace(' Bed', ''))

df['beds'] = df['beds'].replace('-', 0)

df['beds']= df['beds'].astype(int)

df['rent'] = df['rent'].map(lambda x: str(x).replace('$','').replace(',','')).astype('int')

df['baths'] = df['baths'].map(lambda x: x.replace(' Bath', '')).astype('float')

df

"""이로서 rent, beds, baths 열에는 수치형 데이터만 남게됩니다. 다시 데이터 타입을 확인해봅시다."""

df.info()

"""인근 지역을 나타내는 'neighborhood'열에 대해서 카운트하여 출력해봅시다."""

df.groupby('neighborhood')['rent'].count().to_frame('count')\
.sort_values(by='count', ascending=False)

"""특정 키워드에 대해서 카운트를 할 수도 있습니다. 'Upper East Side'가 언급된 데이터는 몇 개가 있는지 카운트해봅시다."""

df[df['neighborhood'].str.contains('Upper East Side')]['neighborhood'].value_counts()

"""각 인근 지역별로 평균 렌트비는 얼마인지 출력해봅시다."""

df.groupby('neighborhood')['rent'].mean().to_frame('mean')\
.sort_values(by='mean', ascending=False)

"""SoHo, Downtown Manhattan, Manhattan이 가장 렌트비가 높고, 렌트비가 가장 낮은 곳은 Prospect Lefferts Gardens, Flatbush, Central Brooklyn, Brooklyn입니다. 이쯤에서 데이터를 실수로 유실하지 않도록 한 번 저장해줍니다."""

df.to_csv('sparta_coding_df.csv', index=False)

df = pd.read_csv('sparta_coding_df.csv', sep=',')

df

!pip install googlemaps

import googlemaps

# gmaps = googlemaps.Client(key='여러분들이 받은 key를 넣으시고 주석 처리를 풀고 실행하세요!')

df.head()

# 3번 인덱스의 위치한 데이터의 'address 열'의 값
df.loc[3, ['address']].values

# 3번 인덱스의 위치한 데이터의 'neighborhood 열'의 값
df.loc[3, ['neighborhood']].values

"""데이터프레임의 3번 인덱스에 위치한 행에 대해서 address 열의 값을 뽑아오고,  
그리고 neighborhood 열에서 가장 마지막으로 언급되는 단어의 값을 뽑아옵니다.
"""

ta = df.loc[3,['address']].values[0] + ' '\
+ df.loc[3,['neighborhood']].values[0].split(', ')[-1]

ta

"""이제 해당 주소를 Google Maps API에 전달하겠습니다."""

geocode_result = gmaps.geocode(ta)

geocode_result

"""여기서는 우편 번호(ZIP)만 추출하려고합니다. ZIP 코드가 어디에 있는지 잘 보세요. ZIP 노드는 이 파일에서 아래와 구조로 추적할 수 있습니다.  

address_componens > types에 postal_code가 있는 구간 > short_name
"""

for piece in geocode_result[0]['address_components']:
    if 'postal_code' in piece['types'] :
        print(piece['short_name'])

"""우리가 원하는 정보를 얻고있는 것 같습니다. 그러나 한 가지 주의 사항이 있습니다. 주소 열을 자세히 살펴보면, (또는 저처럼 여러번 시도해보다가 깨달으실 수도 있는 사실입니다.) 가끔 전체 주소가 제공되지 않는 경우가 있습니다. 이렇게 하면 우편 번호를 얻을 수가 없습니다. 이에 대해서는 nan값. 즉, Null 값(결측값)이 들어가도록 해줍니다.  

함수 내에서 Google Maps Geolocation API로 호출 할 주소를 아까 연습해봤던 것처럼 연결합니다. 정규 표현식(Regular Expression)을 사용하여 거리 번호로 시작하는 호출로만 호출을 제한합니다. 그런 다음 JSON을 분석하여 우편 번호(ZIP)를 얻어냅니다. 우편 번호를 찾으면 반환하고, 그렇지 않으면 np.nan 또는 null 값을 반환합니다.
"""

import re
def get_zip(row):
    try:
        addy = row['address'] + ' ' + row['neighborhood'].split(', ')[-1]
        print(addy)
        if re.match('^\d+\s\w', addy):
            geocode_result = gmaps.geocode(addy)
            for piece in geocode_result[0]['address_components']:
                if 'postal_code' in piece['types']:
                    return piece['short_name']
                else:
                    pass
        else:
            return np.nan
    except:
        return np.nan

"""위에서 re.match는 파이썬에서 제공하는 정규 표현식을 사용한 것입니다.  
정규 표현식에 대해서 다음과 같이 따로 설명하는 자료를 아래 링크로 만들었습니다.  

정규 표현식 이론 설명(현 튜터 블로그) : https://wikidocs.net/21703 (실습에 약 20분 소요)

위 코드에 대한 별도 설명 자료 : https://colab.research.google.com/drive/1MRlw-vP_EOT_u95XSmzBUIqpk9aquNEf?usp=sharing

너무 어렵다면 이런 것도 있구나만 기억하고 넘어갑시다. 저도 필요할 때마다 찾아보고 평소에 굳이 외우고 있지 않습니다.  
정규 표현식은 정말 소수의 분들을 제외하면 모든 분들한테 어렵기 때문입니다 ^^;

DataFrame에서 apply 메소드를 실행합니다. 각 행에 대해서 이 함수들을 모두 적용한다는 의미인데요. 결과적으로 수백 개 이상의 행에 이 함수가 적용된다는 것이므로 실행 시간은 조금 걸립니다.
"""

#@title 기본 제목 텍스트
df['zip'] = df.apply(get_zip, axis=1)

"""모두 처리가 되었습니다. zip 열이 정상적으로 추가되었는지 확인해봅시다."""

df

"""zip 열이 추가된 해당 데이터프레임을 데이터 유실이 없도록 저장합니다."""

df.to_csv('sparta_coding_df_with_zip.csv', index=False)

df = pd.read_csv('sparta_coding_df_with_zip.csv', sep=',', dtype={'zip':object})

df

"""우편 번호(ZIP 코드)를 얻을 수 없는 경우에는 Null 값(결측값)을 넣도록 하였기 때문에, 결측값이 아닌 경우의 데이터는 몇 개인지 확인해봅시다."""

df[df['zip'].notnull()].count()

"""우편 번호(ZIP 코드)인 데이터는 총 940개입니다.

기존 데이터는 1942개지만 이 중 940개의 데이터만 사용하겠습니다.
"""

zdf = df[df['zip'].notnull()].copy()

zdf

# 만약 로드한 데이터의 zip의 열이
# 10003과 같은 형태가 아니라 10003.0 와 같은 소수점 형태로 로드된다면 아래의 코드를 사용해줍니다.
zdf['zip'] = zdf['zip'].str.replace('\.0', '')

zdf

"""우편 번호에 따라서 평균 월세 비용을 정렬해서 출력해봅시다."""

zdf_mean = zdf.groupby('zip')['rent'].mean().to_frame('avg_rent')\
.sort_values(by='avg_rent', ascending=False).reset_index()
zdf_mean

zdf_mean.avg_rent.min()

zdf_mean.avg_rent.max()

"""우편 번호를 기준으로 데이터를 시각화하는 가장 좋은 방법은 데이터를 색상 스펙트럼에 따라서 표현한 히트 맵(heat map)입니다. 패키지 folium을 사용하여 월세에 대한 히트맵을 그려봅시다.  

구글 검색을 통해 ZIP 코드가 연결되어져 있는 뉴욕의 GeoJSON 파일을 찾았습니다. 해당 파일을 다운로드합니다.
"""

!wget https://raw.githubusercontent.com/fedhere/PUI2015_EC/master/mam1612_EC/nyc-zip-code-tabulation-areas-polygons.geojson

"""히트 맵에 사용하고 싶은 열(여기서는 avg_rent) 뿐만 아니라, 히트맵의 키가 되는 열(zip)을 참조하도록 합니다. 저는 히트 맵에 사용하고 싶은 열로 평균 월세를 사용했습니다. 다른 옵션으로 색상 팔레트와 색상을 조정하기 위한 다른 인수를 조정할 수 있습니다.


"""

import folium

m = folium.Map(location=[40.748817, -73.985428], zoom_start=13)

m.choropleth(
    geo_data=open('nyc-zip-code-tabulation-areas-polygons.geojson').read(),
    data=zdf_mean,
    columns=['zip', 'avg_rent'],
    key_on='feature.properties.postalCode',
    fill_color='YlOrRd', fill_opacity=0.7, line_opacity=0.2,
    )

m

"""히트 맵이 완성되면 어느 지역의 월세가 높고 낮은 지 알 수 있습니다. 어떤 지역을 임대할 지 정하는 데 도움이 되기는 하지만, 회귀 모델링을 사용해 좀 더 분석해봅시다."""

zdf.head()

zdf

"""#데이터 모델링

여기서는 침실의 개수가 임대료에 미치는 영향을 알아봅시다. 여기서는 두 개의 패키지를 사용합니다. 첫번째는 statsmodels이고, 두번째는 patsy로 statsmodels의 사용을 더 쉽게 만들어줍니다.
"""

import patsy
import statsmodels.api as sm

"""model변수는 smf의 OLS(최소제곱법)을 사용하여 회귀모형을 만듭니다 formula는 '종속변수 ~ 독립변수1 + 독립변수2 + 독립변수3'과 같이 형식에 맞춰 분석하고자 하는 종속변수(왼쪽)와 독립변수(오른쪽)를 넣으면 됩니다.  

왼편(물결 표시전)은 예측 대상(종속 변수)인 임대료입니다. 오른편은 예측을 위한 변수(독립 변수)인 우편 번호와 침대 개수입니다. 해당 공식은 우편 번호와 침대 개수가 임대료에 어떤 영향을 미치는지 알고 싶다는 것을 의미합니다.  
"""

zdf

"""zip은 더미형 변수다. 원-핫 인코딩이 필요하다. pasty.dmatrices를 사용하면 자동으로 할 수 있다."""

f = 'rent ~ C(zip) + beds'
y, X = patsy.dmatrices(f, zdf, return_type='dataframe')

"""https://kiyoja07.blogspot.com/2019/03/python-linear-regression.html

patsy.dmatirces()에 데이터프레임과 공식을 전달합니다. 아래의 코드를 보면 patsy는 예측 변수에 X 행렬을, 응답 변수에 y 벡터를 설정해 sm.OLS를 입력하고, fit()을 호출해 모델을 실행합니다. 마지막으로 모델의 결과를 출력합니다.  

아래 모델의 결과를 봅시다. 일단 관측을 한 데이터의 개수(No. Observations)는 940개입니다. 조정 R2(adjusted R2 / 테이블상으로는 Adj. R-squared)는 0.487입니다. **R-squared**는 앞서 회귀분석을 실시한 "임대료  = 우편번호 + 방의 개수 * weight"라는 모델식의 적합성을 말해줍니다. 결과는 선형 회귀분석 모델이 임대료의 변동성의 48.7%를 설명한다는 의미이다. R-squared는 0 ~ 1의 값을 가지고 0 이면 모델의 설명력이 전혀 없는 상태, 1이면 모델이 완벽하게 데이터를 설명해주는 상태이다. 사회과학에서는 보통 0.4 이상 이면 괜찮은 모델이라고 판단합니다.

 중요한 것은 F-통계 확률이 2.64e-94라는 것입니다. 회귀모형에 대한 (통계적) 유의미성 검증 결과, 유의미함 (p < 0.05) 
이것이 왜 중요하냐면 침대 개수와 우편번호만을 사용해서 해당 모델이 제3의 임대료의 편차를 설명할 수 있다는 것을 의미합니다. 좋은 결과일까요? 더 좋은 답을 찾기 위해 결과의 중간 부분을 살펴봅시다.  

아래 테이블 형태의 출력 부분의 각 열은 모델에서 각 독립 변수의 정보를 제공합니다. 왼쪽에서 오른쪽으로 다음 정보를 확인할 수 있습니다. 변수의 모델과의 계수, 표준 오차(standard error), t-통계, t-통계의 P값, 95% 신뢰 구간입니다.  

이 값들이 무엇을 말할까요? P 값의 열을 보면 독립 변수들의 통계적 유의성(statistically significnat)를 결정할 수 있습니다. 회귀 모델에서의 통계적 유의성은 독립 변수와 응답 변수간의 관계가 우연히 발생하지는 않았다는 것을 의미합니다. 일반적으로 통계학자들은 P 값이 .05인 것을 기준으로 결정합니다. P갑이 .05라는 것은 해당 결과가 우연히 발생할 확률이 5%뿐이라는 것입니다. 침실의 개수는 여기서 확실히 유의성을 가집니다.
"""



y

results = sm.OLS(y, X).fit()
results.summary()

print(results.summary())

"""* Adj.R-squred : 보통 설명력이라고 말하는 값인데 주어진 데이터를 현재 모형이 얼마나 잘 설명하고 있는지를 나타내는 지수입니다.  

* Prob(F-statistics) : 모형에 대한 p-value 로 통상 0.05이하인 경우 통계적으로 유의하다고 판단합니다. 2.64e-94는 2.64 * 10^-94를 의미하므로 유의합니다.  
* P>[t] : 각 독립변수의 계수에 대한 p-value로 해당 독립변수가 유의미한지 판단합니다. 0.05 이하인 경우 통계적으로 유의미하다고 판단하며, 0.05보다 큰 경우에는 해당 변수는 통계적으로 유의미하지 않습니다. (예측에 별 도움이 안 되는 변수라는 의미입니다.)
"""
