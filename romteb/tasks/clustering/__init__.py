"""Romanian clustering tasks.

Empty for now. SIB200ClusteringS2S is reused from MTEB and does not need
a class here. The native RORetrieval outlet / news-type clustering
modules referenced by ``romteb.eval_config`` do not exist yet; their
imports were dropped so this subpackage can be imported cleanly. Add
``ro_news_outlet.py`` / ``ro_news_type.py`` (and re-export their
classes) once the prep script is checked in.
"""

__all__: list[str] = []
