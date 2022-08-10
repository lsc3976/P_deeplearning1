# 3. Linear Regression (선형 회귀)

 
<br><br>

## 선형 회귀(Linear Regression)란?

머신 러닝의 가장 큰 목적은 실제 데이터를 바탕으로 모델을 생성하고 <br>
다른 특정 입력 값을 넣었을때 발생할 아웃풋을 예측하는데에 있다. <br>
선형 회귀(Linear Regression)는 직관적이고 데이터의 경향성을 가장 잘 설명하는 하나의 직선을 예측하는 지도학습 기법이다. <br>
독립변수와 종속변수 데이터를 주고 모델을 트레이닝시켜서 정확도를 높이고, <br>
주어진 독립변수 X에 해당하는 실제 값으로 타겟 Y(종속변수)를 예측할때 이 회귀식의 계수(입력 피처)들이 선형조합으로 표현된다. <br>


<br><BR>

## 선형 회귀의 정의
  
  
  
  
  ### 1) 단순 선형 회귀
  

  
  

  <img src='https://github.com/lsc3976/P_deeplearning1/blob/main/image/10.png?raw=true' /><br>
  (일반적인 단순 선형회귀)
  <br><br>
  
  예를 들어 집의 크기(독립 변수)를 사용해서 주택 가격(종속 변수)을 예측할때
  
  $Y$(price) $= w_0 + w_1 * X$(size)
  
  라는 1차 함수식(회귀식)으로 모델링 할수있다.
  이때 기울기 $w_1$과 절편인 $w_0$을 회귀 계수(Regression coefficients)로 지칭한다. <br>
  위와 같은 1차 함수로 모델링했다면 실제 주택 가격은 이러한 1차 함수 값에서 실제 값만큼의 오류 값을 빼거나 더한 값이 된다.
  
  이렇게 실제값과 회귀 모델의 차이에 따른 오류값을 남은 오류, 즉 잔차(residual)라 부른다. <br>
  __최적의 회귀 모델을 만든다는 것__ 은 전체 데이터의 잔차(오류값)합이 최소가 되는 모델을 만든다는 의미이며, <br>
  동시에 오류값 합이 최소가 될 수 있는 __최적의 회귀 계수를 찾는다__ 는 의미이다.
  <br><br>
  
  ### 2) 다중 선형 회귀

  사실 주택 가격은 단순히 집의 크기(하나의 독립변수)만으로 결정되지않고 <br>
  방의 개수, 주변 교통수단과의 거리 등 다양한 요소로부터 영향을 받는다. <br>
  

  $Y = w_1X_1 + w_2X_2 + w_3X_3 +  ...  + w_nX_n + b$

  이러한 다수의 독립 변수를 가지고 주택 가격을 예측할때 Y(종속변수)는 여전히 1개 이지만 <br>
  X(독립변수)는 여러개가 된다. 이것을 다중 선형 회귀 분석이라고 한다.
  <br><br><br>
  
  
## 손실 함수

<br>
<img src='/image/117.png' /><br>
 
```
손실 함수 : 실제 y 값에 비해 가정한 모델 h(x))(추정값)이 얼마나 잘 예측했는지 판단하는 함수이다.
```

회귀식을 모델링 할 경우 당연히 실제 데이터와 회귀식에는 오차가 발생 한다. <br>
선형회귀는 위에 언급된 오류값(잔차)의 합이 최소가 되는 최적 회귀 계수를 찾는 것 <br>
(가장 정확한 예측선을 긋는 것)이 궁극적인 목표가 된다.<br>

* 어떠한 문제를 해결하고자 하는가에 따라 적합한 손실함수를 설정해야 좋은 결과를 얻을 수 있다. <br>

 ### 비용 함수
 
 머신러닝 알고리즘에서 최적화는 비용함수의 값이 가장 작아지는 최적의 파라미터를 찾는 과정을 말한다. <br>
 이를 달성하기 위해서, 경사하강법(Gradient Descent) 기반의 방식이 가장 기본이 되는 알고리즘이다. <br>

 목적함수란, 최적화 알고리즘에서 최대값 또는 최소값을 찾아야하는 함수를 말한다. <br>
 즉, 비용함수는 최적화 알고리즘의 목적함수이다. <br>

 
회귀 모델에 쓰이는 비용함수에는 MSE, MAE, RMES 등이 있으며 <br>
분류 모델에 쓰이는 비용함수에는 Binary cross-entropy, Categorical cross-entropy 등이 있다. 
<br><br>
 
 <img src="/image/116.png"> <br>
평균 제곱 오차 (Mean Squre Error, MSE)의 경우<br>
n은 총 데이터, y는 출력결과이고 t는 실제 데이터의 값이다. <br>
출력결과 - 실제데이터의 차이 를 제곱(음수가 나오는걸 방지)한뒤 평균을 내어 표현하고있다. <br>
이와같이 비용함수는 손실함수를 사용하여 정의될 수 있다.

 
<br><br><br>


  
## 비선형 회귀(Non-linear Regression)란?
  

  <br>
<img src='https://github.com/lsc3976/P_deeplearning1/blob/main/image/12.png?raw=true' /><br>

비선형 모델은 입력되는 데이터(독립 변수, 종속 변수)를 어떻게 변형 하더라도 <br>
회귀 계수를 선형 결합식으로 표현할 수 없는 모델을 말한다. <br>

선형 회귀 모델은 파라미터(회귀 계수)에 대한 해석이 단순하지만 <br>
비선형 회귀 모델은 모델의 형태가 복잡할 경우 해석이 매우 어려워진다. <br>
그래서 보통 모델의 해석을 중시하는 통계 모델링에서는 비선형 회귀 모델을 잘 사용하지 않는다. <br>
하지만 회귀 모델의 목적이 해석이 아니라 결과값의 예측에 있다면 비선형 모델은 대단히 유연하기 때문에 <br>
복잡한 패턴을 갖는 데이터에 대해서도 모델링이 가능하고 충분히 많은 데이터를 갖고 있어서 오류값을 줄일수있고 <br>
예측 자체가 목적인 경우에 비선형모델은 매우 뛰어난 도구가 된다.
  
<br><br><br>
 

## 활성화 함수
  
  <br>
  
  
  
### 뉴런에서 해답 가져오기
  
  
  <img src='https://github.com/lsc3976/P_deeplearning1/blob/main/image/133.png?raw=true' /><br>
  인간은 컴퓨터와 달리 목소리, 사진, 언어 와 같은 Unstructrued Data를 쉽게 이해하고 활용 가능하다. <br>
  그것은 뇌의 뉴런, 수상돌기로부터 특정 정보 $x_n$를 받아들이고 시텝틱 가중치 $w_i$를 적용하는 일련의 과정을 거치기때문이다.<br>
  이 가중치는 입력에 얼마나 반응해야하는지를 정의한다. ( $x_n$ $w_i$를 통해 활성화)<br>
  이 모든 값들은 뉴런 핵에서  $y=\sum_i x_i wi+b$ 로 통합되고 일정 수준을 넘어서면 활성화되어 해당 정보를 축삭(axon)으로 보내져서<br>
  다른 프로세스를 거치는데, 일반적으로 σ(y)를 통해서 비선형처리가 된다. 이 후 최종 목적지를 거쳐 다른 뉴런으로 보내진다. <br>
  이처럼, 인간의 뇌가 작동하는 방식을 모방해 비선형 회귀 모델을 구축하면 놀랍게도 컴퓨터가 마치 인간처럼<br>
  구조화되지않은 자료(보이스,사진,언어)를 이해하고 활용할수 있게된다. 이것이 비선형 회귀의 강력한 핵심이다.
<br><br><br>
  
  

  ### 활성화 함수의 정의


  <img src='https://github.com/lsc3976/P_deeplearning1/blob/main/image/120.png?raw=true' /><br>
  뉴런처럼 입력 신호의 총합을 출력 신호로 변환하는 함수를 일반적으로 활성화 함수라고 한다. <br>
  입력 신호의 총합이 활성화를 일으키는지를 정하는 역할이다.<br>
  위 그림은 신경망 그림으로 가중치(w)가 달린 입력 신호(x)와 편향(b)의 총합을 계산하고<br>
  그 다음에 함수 f에 넣어 출력하는 흐름을 보여준다.<br>
 
  <br><br>

  <br>
  
  ### 1) 시그모이드(Sigmoid) 함수

  <br><br>
  <img src='https://github.com/lsc3976/P_deeplearning1/blob/main/image/778.png?raw=true' /><br>
  
  $sigmoid(x) = \frac{1}{1 +e^-x}$ <br>
  
  시그모이드 함수는 Logistic 함수라고 불리기도 하며, <br>
  x의 값에 따라 0~1의 값을 출력하는 S자형 함수이다. <br>
 

 
 
  
 
  <br>
  



  ### 2) ReLU (Rectified Linear Unit) 함수
<br>
  <img src='https://github.com/lsc3976/P_deeplearning1/blob/main/image/783.png?raw=true' /><br>
  레이어의 층이 깊어질수록 내부 은닉층(hidden layer)을 활성화 시키는 함수로 <br>
  ReLU라는 활성화 함수를 사용하게 되는데, 이 함수는 쉽게 말해 0보다 작은 값이 나온 경우 0을 반환하고, <br>
  0보다 큰 값이 나온 경우 그 값을 그대로 반환하는 함수다. 0보다 큰 값일 경우 1을 반환하는 시그모이드와 다르다. <br>
  <br>

  시그모이드와 ReLU 이외에도 여러가지 활성화 함수가 있다. <br><br>

  <br><br><br>
  
  사진 및 참고자료 출처 : [출처1](https://ko.d2l.ai/chapter_deep-learning-basics/linear-regression.html) [출처2](https://needjarvis.tistory.com/564) [출처4](http://www.gisdeveloper.co.kr/?p=8395) [출처5](https://076923.github.io/posts/Python-pytorch-4/) [출처6](https://velog.io/@hh3990/%EC%84%A0%ED%98%95%ED%9A%8C%EA%B7%80Linear-Regression) [출처7](https://medium.com/@kmkgabia/ml-sigmoid-%EB%8C%80%EC%8B%A0-relu-%EC%83%81%ED%99%A9%EC%97%90-%EB%A7%9E%EB%8A%94-%ED%99%9C%EC%84%B1%ED%99%94-%ED%95%A8%EC%88%98-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0-c65f620ad6fd
) [출처8](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=handuelly&logNo=221824080339) [출처9](https://happy-obok.tistory.com/55)
  
