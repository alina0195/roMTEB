"""Romanian clustering tasks.

SIB200ClusteringS2S is reused from MTEB. Native RORetrieval clustering
(outlet + content type) is registered once the prep script has been run
with a local ``--input`` path.
"""

from romteb.tasks.clustering.ro_news_outlet import RoNewsOutletClusteringP2P
from romteb.tasks.clustering.ro_news_type import RoNewsTypeClusteringP2P

__all__ = [
    "RoNewsOutletClusteringP2P",
    "RoNewsTypeClusteringP2P",
]
