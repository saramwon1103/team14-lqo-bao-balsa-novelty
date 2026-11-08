/*+ MergeJoin(k mk mi_idx it t)
 MergeJoin(mk mi_idx it t)
 NestLoop(mi_idx it t)
 NestLoop(mi_idx it)
 IndexScan(k)
 SeqScan(mk)
 SeqScan(mi_idx)
 IndexScan(it)
 IndexScan(t)
 Leading((k (mk ((mi_idx it) t)))) */
SELECT MIN(mi_idx.info) AS rating,
       MIN(t.title) AS movie_title
FROM info_type AS it,
     keyword AS k,
     movie_info_idx AS mi_idx,
     movie_keyword AS mk,
     title AS t
WHERE it.info ='rating'
  AND k.keyword LIKE '%sequel%'
  AND mi_idx.info > '9.0'
  AND t.production_year > 2010
  AND t.id = mi_idx.movie_id
  AND t.id = mk.movie_id
  AND mk.movie_id = mi_idx.movie_id
  AND k.id = mk.keyword_id
  AND it.id = mi_idx.info_type_id;

