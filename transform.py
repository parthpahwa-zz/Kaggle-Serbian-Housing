import numpy as np
import pandas as pd
import time
import datetime
from math import log,sqrt
area = []

def convOwnr(x):
	if x == 'Investment':
		return 1
	return 0

def convTime(x):
	if (x!=x):
		return x
	x = str(int(x))
	if(len(x) < 4):
		return float('NaN')
	return int(x)

def convTimestamp(x):
	try:
		y = time.mktime(datetime.datetime.strptime(x, "%Y-%m-%d").timetuple())
	except Exception,e:
		print e , x
		return float('NaN')
	return y


def convArea(x):
	for i in range (0,len(area)):
		if(x == area[i]):
			return i

def encodeSubArea(df):
	df['sub_area1'] = df['sub_area'].apply(lambda x: convArea(x))
	return df

def getYear(x):
	return int(x.split('-')[0])

def convUNIXtime(df):
	df['build_year'] = df.build_year.apply(lambda x: convTime(x))
	df['sellYear'] = df.timestamp.apply(lambda x: getYear(x))
	df.timestampConv = df.timestamp.apply(lambda x: convTimestamp(x))
	return df

def encodeOwnr(df):
	df['newOwnr'] = df['product_type'].apply(lambda x: convOwnr(x))
	return df

def fillBuildTime(df):
	temp = df[['newTIME','newOwnr','sub_area']].groupby(['newOwnr','sub_area']).mean()
	index = df['newTIME'].index[df['newTIME'].apply(np.isnan)]
	for indx in index:
		sr = df.ix[indx] 
		df["newTIME"].ix[indx] = temp.loc[sr['newOwnr'],sr['sub_area']][0]
	df = finalBuildTime(df)
	return df


		
def finalBuildTime(df):
	temp = df[['newTIME','newOwnr','sub_area']].groupby(['newOwnr','sub_area']).mean()
	index =  df['newTIME'].index[df['newTIME'].apply(np.isnan)]
	for indx in index:
		sr = df.ix[indx] 
		if(sr['newOwnr'] == 1):
			df['newTIME'].ix[indx] = temp.loc[0,sr['sub_area']][0]
		else :
			df['newTIME'].ix[indx] = temp.loc[1,sr['sub_area']][0]
	return df

def fillFloor(df):
	index =  df['floor'].index[df['floor'].apply(np.isnan)]
	inc = 387275400
	minVal = -2305730000
	maxVal = 94513120000
	tempTimeLst =  np.linspace(-2305730000,94513120000,251)
	temp = df[['floor','newTIME','newOwnr']].groupby([pd.cut(df['newTIME'],tempTimeLst),'newOwnr']).mean()
	for indx in index:
		sr = df.ix[indx] 
		
		val = sr['newTIME'] 
		val = int((val - minVal)/inc)
		timeString = '('+str("%d" %(minVal + (val)*inc))+', '+str("%d" %(minVal + (val+1)*inc))+']'
		
		print indx , sr['newTIME'], timeString 
		val = sr['newOwnr']
		df['floor'].ix[indx] = temp.loc[timeString,val][0]
	return df 

def fillMaxFloor(df):
	index=  df['max_floor'].index[df['max_floor'].apply(np.isnan)]

	inc = 387275400
	minVal = -2305730000
	maxVal = 94513120000
	tempTimeLst =  np.linspace(-2305730000,94513120000,251)
	tempFloorLst = np.linspace(0,102,18)
	temp =  df[['max_floor','newTIME','floor']].groupby([pd.cut(df['newTIME'],tempTimeLst),pd.cut(df['floor'],tempFloorLst)]).mean()
	
	for indx in index:
		
		sr = df.ix[indx] 
		
		val = sr['newTIME'] 
		val = int((val - minVal)/inc)
		timeString = '('+str("%d" %(minVal + (val)*inc))+', '+str("%d" %(minVal + (val+1)*inc))+']'
		print indx , sr['newTIME'], timeString 
		
		val = sr['floor'] 
		val = int((val - 0)/6)
		floorString = '('+str("%d" %((val)*6))+', '+str("%d" %((val+1)*6))+']'
		print sr['floor'], floorString
		floorValue = temp.loc[timeString,floorString][0]
		if(floorValue >=  sr['floor'] ) :
			df['max_floor'].ix[indx] = floorValue
		else :
			df['max_floor'].ix[indx] = sr['floor'] + 2

	return df	
	
def encodeYesNo(x):
	if(x == 'yes'):
		return 1
	elif(x == 'no'):
		return 0
	return -1
def encodeEcology(x):
	if(x == 'no data'):
		return 0
	elif(x == 'poor'):
		return 1
	elif(x == 'satisfactory'):
		return 2
	elif(x == 'good'):
		return 3
	elif(x == 'excellent'):
		return 4
	print 'ERROR  ',x

def encode(df):
	encodeList = ['culture_objects_top_25','thermal_power_plant_raion','incineration_raion','oil_chemistry_raion','radiation_raion','railroad_terminal_raion','big_market_raion','nuclear_reactor_raion','detention_facility_raion','water_1line','big_road1_1line','railroad_1line']
	for col in encodeList:
		df[col] = df[col].apply(lambda x: encodeYesNo(x))
	df['ecology'] = df['ecology'].apply(lambda x: encodeEcology(x))
	return df

def dropCols(df):
	dropList = ['material','build_year','kitch_sq','product_type','sub_area']
	df = df.drop(dropList,axis = 1)
	return df

def predictXGB(lst, df, col):
	print "Current col: ",col
	xTest = []
	xTrain = []
	yTrain = []
	
	index =  df[col].index[df[col].notnull()]
	for indx in index:
		temp = []
		for val in lst:
			temp.append(df[val].ix[indx])
		xTrain.append(temp)
		yTrain.append(df[col].ix[indx])

	index =  df[col].index[df[col].apply(np.isnan)]	
	for indx in index:
		temp = []
		for val in lst:
			temp.append(df[val].ix[indx])
		xTest.append(temp)

	xTrain = pd.DataFrame(xTrain)
	xTest = pd.DataFrame(xTest)
	yTrain = np.array(yTrain)

	xgb_model = xgb.XGBRegressor()
	clf = GridSearchCV(xgb_model, {'max_depth': [4],'n_estimators': [400]}, verbose=1,n_jobs=1)
	clf .fit(xTrain, yTrain)
	predictions = clf.predict(xTest)
	i = 0
	for indx in index:
		df[col].ix[indx] = predictions[i] 
		i += 1
	return df

def getFinalLst(lst,emptyCols,col,val):
	temp = []
	flag = 1
	for item in lst:
		if item[1] not in emptyCols and item[1] not in ['price_doc','id']:
			if item [0] > 0.3 - val:
				temp.append(item[1])
	if len(temp) < 7 :
		flag = 0
	return temp,flag

def predictNULL(df,val):
	emptyCols = []
	for col in df.columns.values:
		# print col
		x = len(df[col].index[df[col].apply(np.isnan)])
		if x > 0:
			emptyCols.append(col)
	while 1:
		lenE = len(emptyCols)
		for col in emptyCols:
			lst = []
			for indx in df.columns.values:
				lst.append([indx, abs(df[col].corr(df[indx]))])
			lst = sorted(zip(map(lambda x: round(x, 8), [row[1] for row in lst ]),[row[0] for row in lst ]), reverse=True)
			lst , flag = getFinalLst(lst,emptyCols,col,val)
			# lst.append('price_doc')
			if flag:
				df = predictXGB(lst, df,col)
				emptyCols.remove(col)
			else :	
				print col , len(lst)
				pass
		if val != 0:
			val -= 0.02
		if len(emptyCols) == lenE or len(emptyCols) == 0:
			break
	return df

def transfromDF(df):
	df = encodeSubArea(df)
	df = convUNIXtime(df)
	df = encodeOwnr(df)
	df = encode(df)
	df = df.drop(['sub_area','product_type'],axis = 1)
	return df

df = pd.read_csv("train.csv")
area = list(df.sub_area.unique())
df = transfromDF(df)

df.to_csv("allEncoded.csv",index =False)
