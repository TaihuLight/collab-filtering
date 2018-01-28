#get_ipython().run_line_magic('reload_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
#get_ipython().run_line_magic('matplotlib', 'inline')

from fastai.learner import *
from fastai.column_data import *

path='data/ml-latest-small/'

ratings = pd.read_csv(path+'ratings.csv')
ratings.head()


movies = pd.read_csv(path+"movies.csv")
movies.head()


g = ratings.groupby('userId')['rating'].count()
topUsers = g.sort_values(ascending=False)[:15]

g = ratings.groupby('movieId')['rating'].count()
topMovies = g.sort_values(ascending=False)[:15]

top_r = ratings.join(topUsers, rsuffix = '_r', how = 'inner', on = 'userId')
top_r = top_r.join(topMovies,rsuffix='_r', how = 'inner', on = 'movieId')

pd.crosstab(top_r.userId,top_r.movieId, top_r.rating,aggfunc=np.sum)

val_idxs = get_cv_idxs(len(ratings))
wd = 2e-4
n_factors=50

cf = CollabFilterDataset.from_csv(path, 'ratings.csv','userId', 'movieId','rating')
learn = cf.get_learner(n_factors, val_idxs, 64, opt_fn = optim.Adam)

learn.fit(1e-2, 2, wds = wd, cycle_len=2, cycle_mult= 2, use_wd_sched= True)

preds = learn.predict()

y = learn.data.val_y
sns.jointplot(preds, y, kind='hex', stat_func=None)


class DotProduct(nn.Module):
    def forward(self, u, m):
        return (u*m).sum(1)

model = DotProduct()

model(a,b)


u_uniq = ratings.userId.unique()
user2idx = {o:i for i,o in enumerate(u_uniq)}
ratings.userId = ratings.userId.apply(lambda x: user2idx[x])

m_uniq = ratings.movieId.unique()
movie2idx = {o:i for i,o in enumerate(m_uniq)}
ratings.movieId = ratings.movieId.apply(lambda x: movie2idx[x])

n_users=int(ratings.userId.nunique())
n_movies=int(ratings.movieId.nunique())


n_users=int(ratings.userId.nunique())
n_movies=int(ratings.movieId.nunique())

class EmbeddingDot(nn.Module):
    def __init__(self,n_users, n_movies):
        super().__init__()
        self.u = nn.Embedding(n_users, n_factors)
        self.m = nn.Embedding(n_movies, n_factors)
        self.u.weight.data.uniform_(0,0.05)
        self.m.weight.data.uniform_(0,0.05)
    
    def forward(self,cats,conts):
        users, movies = cats[:,0], cats[:,1]
        u,m = self.u(users),self.m(movies)
        return (u*m).sum(1)

x = ratings.drop(['rating', 'timestamp'],axis=1)
y = ratings['rating'].astype(np.float32)


data = ColumnarModelData.from_data_frame(path, val_idxs, x, y, ['userId', 'movieId'], 64)

wd = 1e-5
model = EmbeddingDot(n_users, n_movies)
opt = optim.SGD(model.parameters(), 1e-1, weight_decay=wd, momentum=0.9)

fit(model, data, 3, opt, F.mse_loss)

set_lrs(opt, 0.01)

fit(model, data, 3, opt, F.mse_loss)

min_rating, max_rating = ratings.rating.min(), ratings.rating.min()
min_rating, max_rating

def get_emb(ni,nf):
    e = nn.Embedding(ni,nf)
    e.weight.data.uniform_(-0.01, 0.01)
    return e

class EmbeddingDotBias(nn.Module):
    def __init__(self, n_users, n_movies):
        super().__init__()
        (self.u,self.m,self.ub,self.mb) = [get_emb(*o) for o in [
            (n_users,n_factors), (n_movies,n_factors), (n_users,1), (n_movies,1)
        ]]
    
    def forward(self, cats, conts):
        users, movies = cats[:,0], cats[:,1]
        um = (self.u(users)*self.m(movies)).sum(1)
        res = um + self.ub(users).squeeze() + self.mb(movies).squeeze()
        res = F.sigmoid(res)*(max_rating-min_rating)+min_rating
        return res
    

wd = 2e-4
models = EmbeddingDotBias(cf.n_users,cf.n_items )
opt = optim.SGD(model.parameters(), 1e-1, weight_decay = wd, momentum=0.9)

fit(model, data, 3, opt, F.mse_loss)

