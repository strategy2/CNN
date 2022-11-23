import os
import matplotlib.pyplot as plt
import pandas as pd
import talib
from ta import add_all_ta_features
from ta.utils import dropna
from datetime import datetime, timedelta
from PIL import Image


from pyts.multivariate.image import JointRecurrencePlot
from pyts.datasets import load_basic_motions
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import GramianAngularField
from pyts.datasets import load_gunpoint


class ConvertTicker2Image:

    def __init__(self):
        self.basepath='/home/hoku/Desktop/AI_TS/BACKTESTING/1_min/2022'
        self.imagepath='/home/hoku/Desktop/time_series_to_image/images'


    def create_images(self,percent=0.02,ndays=2):
        ''' creates images of last n days of market action, utilizing the pyts library
         :param percent (float) - percent rise in intraday prices to select as positive ground truth
         :param ndays (int) - number of days to include in the image to predict the following day ground truth,
         '''

        #iterate over tickers
        for ticker in sorted(os.listdir(self.basepath)):
            try:

                # for each ticker, create dataset for a whole year
                print("creating dataset for 1 year for ticker {}".format(ticker))
                df=self.create_year_data(ticker=ticker)
                df.index=df['time']
                df.index = pd.to_datetime(df.index)
                df['date']=df.index.strftime("%Y-%m-%d")
                df=df.drop(['time'],axis=1)

                #Iterate over dates to create images for each date looking back n days
                dates=sorted(list(set(df.index.strftime("%Y-%m-%d"))))  #note starting at date3
                for dt in dates[ndays:]: #start n days in

                    #create dataset that represents today
                    df_date=df.loc[df['date']==dt,:]

                    #go back n days to get the start date (dts=start date)
                    dts=dates[dates.index(dt)-ndays]

                    #select data from n days prior to current day (currently continuous)
                    start=datetime(int(dts.split('-')[0]),int(dts.split('-')[1]),int(dts.split('-')[2]), 9, 30, 0)
                    end = datetime(int(dt.split('-')[0]), int(dt.split('-')[1]), int(dt.split('-')[2]), 9, 0, 0)
                    mask = (df.index > start) & (df.index <= end)
                    d4i=df.loc[mask]
                    d4i=d4i.drop(['date'],axis=1)

                    #create image from data
                    d4i_n=d4i.to_numpy()
                    d4i_n=d4i_n.reshape(1,d4i_n.shape[0],d4i_n.shape[1])
                    image=self.recplot(X=d4i_n)
                    im = Image.fromarray((image* 255).astype('uint8'), mode='L')
                    im=im.resize((224,224))


                    #find ground truth using data from current day (df_date)
                    amount=0; label=0
                    df_trading=df_date.between_time('9:30','16:00')
                    if (df_trading['close'][-1]-df_trading['close'][0])>df_trading['close'][0]*percent:
                        #amount=((df_trading['close'][-1]-df_trading['close'][0])/df_trading['close'])
                        #print(amount)
                        label=1
                        print('for ticker {} on date {} rise of more than {} percent'.format(ticker,dt,percent))

                    # setup saving directory
                    if not os.path.exists(os.path.join(self.imagepath,dt)):
                        os.mkdir(os.path.join(self.imagepath,dt))
                    if not os.path.exists(os.path.join(self.imagepath,dt,str(label))):
                        os.mkdir(os.path.join(self.imagepath,dt,str(label)))

                    im.save(os.path.join(self.imagepath,dt,str(label),ticker+'.jpeg'))
            except:
                print("problem with ticker {}".format(ticker))




    def create_year_data(self,ticker='AAPL'):
        ''' for a given ticker, get all the data saved as month data and concatenate to a year
        :param ticker (str) - the ticker in symbolic format
        '''

        df=pd.DataFrame(columns=['time','volume','open','close','high','low'])
        for month in sorted(os.listdir(os.path.join(self.basepath,ticker))):
            md=pd.read_csv(os.path.join(self.basepath,ticker,month))
            df=pd.concat([df,md])

        df=df[['time','volume','open','close','high','low']]
        df=df.sort_values(by='time')
        df = add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
        df=df.drop(['volatility_bbli','volatility_bbli','volatility_kchi','volatility_kcli','trend_psar_up_indicator','trend_psar_down_indicator'],axis=1)
        df=df.iloc[40:,:]

        #df.to_csv(os.path.join(self.imagepath,'2022_'+ticker+'.csv'))

        return df


    def recplot(self,X):
        ''' Small function wrapping the recurrence plot transformation
        "A joint recurrence plot is an extension of recurrence plots for multivariate time series:
        it is the Hadamard of the recurrence plots obtained for each feature of the multivariate time series."
        (Romano et al, Phyics Letters A Volume 330, Issues 3–4, 20 September 2004, Pages 214-223)
        :param X (nparray): input numpy array with shape (1,X,Y)
        '''

        # Recurrence plot transformation
        jrp = JointRecurrencePlot(threshold='point', percentage=50)
        X_jrp = jrp.fit_transform(X)
        return(X_jrp[0])


if __name__=='__main__':
    c=ConvertTicker2Image()
    c.create_images()