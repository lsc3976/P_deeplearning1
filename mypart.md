# 3. Linear Regression (선형 회귀)


<br><br>

## 선형 회귀(Linear Regression)란?

머신 러닝의 가장 큰 목적은 실제 데이터를 바탕으로 모델을 생성하고 <br>
다른 특정 입력 값을 넣었을때 발생할 아웃풋을 예측하는데에 있다. <br>
선형 회귀(Linear Regression)는 직관적이고 데이터의 경향성을 가장 잘 설명하는 하나의 직선을 예측하는 지도학습 기법이다. <br>
주어진 독립변수 X에 해당하는 실제 값으로 타겟 Y를 예측할때 이 회귀식의 계수(입력 피처)들이 선형 조합으로 표현된다. <br>


<br><BR>

## 선형 회귀의 정의
  
  
  ### 1) 단순 선형 회귀
  
  예를 들어 집의 크기(독립 변수)를 사용해서 주택 가격(종속 변수)을 예측할때
  
  $Y$(price) $= w0 + w1 * X$(size)
  
  ![ima1.png](data:[https://drive.google.com/file/d/1u1PrccqwL9089Qiy-BonTXnC9kWfR2go/view?usp=sharing](https://drive.google.com/file/d/1y46M6PGvkEk6Ieff778jjoLr857uBpDc/view?usp=sharing))
  라는 1차 함수식(회귀식)으로 모델링 할수있다.
  이때 기울기 $w1$과 절편인 $w0$을 회귀 계수(Regression coefficients)로 지칭한다.
  위와 같은 1차 함수로 모델링했다면 실제 주택 가격은 이러한 1차 함수 값에서 실제 값만큼의 오류 값을 빼거나 더한 값이 된다.


  
  
  
