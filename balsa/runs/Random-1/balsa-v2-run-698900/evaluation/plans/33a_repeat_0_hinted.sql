/*+ HashJoin(it1 cn2 mi_idx1 mc2 it2 cn1 kt1 mi_idx2 mc1 kt2 t2 t1 ml lt)
 MergeJoin(cn2 mi_idx1 mc2 it2 cn1 kt1 mi_idx2 mc1 kt2 t2 t1 ml lt)
 MergeJoin(mi_idx1 mc2 it2 cn1 kt1 mi_idx2 mc1 kt2 t2 t1 ml lt)
 HashJoin(mc2 it2 cn1 kt1 mi_idx2 mc1 kt2 t2 t1 ml lt)
 HashJoin(it2 cn1 kt1 mi_idx2 mc1 kt2 t2 t1 ml lt)
 HashJoin(it2 cn1 kt1 mi_idx2 mc1 kt2 t2 t1 ml)
 HashJoin(cn1 kt1 mi_idx2 mc1 kt2 t2 t1 ml)
 HashJoin(kt1 mi_idx2 mc1 kt2 t2 t1 ml)
 HashJoin(mi_idx2 mc1 kt2 t2 t1 ml)
 MergeJoin(mc1 kt2 t2 t1 ml)
 HashJoin(kt2 t2 t1 ml)
 HashJoin(t2 t1 ml)
 HashJoin(t1 ml)
 SeqScan(it1)
 IndexScan(cn2)
 SeqScan(mi_idx1)
 SeqScan(mc2)
 IndexScan(it2)
 SeqScan(cn1)
 IndexScan(kt1)
 IndexScan(mi_idx2)
 IndexScan(mc1)
 IndexScan(kt2)
 SeqScan(t2)
 IndexScan(t1)
 SeqScan(ml)
 IndexScan(lt)
 Leading((it1 (cn2 (mi_idx1 (mc2 ((it2 (cn1 (kt1 (mi_idx2 (mc1 (kt2 (t2 (t1 ml)))))))) lt)))))) */
SELECT MIN(cn1.name) AS first_company,
       MIN(cn2.name) AS second_company,
       MIN(mi_idx1.info) AS first_rating,
       MIN(mi_idx2.info) AS second_rating,
       MIN(t1.title) AS first_movie,
       MIN(t2.title) AS second_movie
FROM company_name AS cn1,
     company_name AS cn2,
     info_type AS it1,
     info_type AS it2,
     kind_type AS kt1,
     kind_type AS kt2,
     link_type AS lt,
     movie_companies AS mc1,
     movie_companies AS mc2,
     movie_info_idx AS mi_idx1,
     movie_info_idx AS mi_idx2,
     movie_link AS ml,
     title AS t1,
     title AS t2
WHERE cn1.country_code = '[us]'
  AND it1.info = 'rating'
  AND it2.info = 'rating'
  AND kt1.kind IN ('tv series')
  AND kt2.kind IN ('tv series')
  AND lt.link IN ('sequel',
                  'follows',
                  'followed by')
  AND mi_idx2.info < '3.0'
  AND t2.production_year BETWEEN 2005 AND 2008
  AND lt.id = ml.link_type_id
  AND t1.id = ml.movie_id
  AND t2.id = ml.linked_movie_id
  AND it1.id = mi_idx1.info_type_id
  AND t1.id = mi_idx1.movie_id
  AND kt1.id = t1.kind_id
  AND cn1.id = mc1.company_id
  AND t1.id = mc1.movie_id
  AND ml.movie_id = mi_idx1.movie_id
  AND ml.movie_id = mc1.movie_id
  AND mi_idx1.movie_id = mc1.movie_id
  AND it2.id = mi_idx2.info_type_id
  AND t2.id = mi_idx2.movie_id
  AND kt2.id = t2.kind_id
  AND cn2.id = mc2.company_id
  AND t2.id = mc2.movie_id
  AND ml.linked_movie_id = mi_idx2.movie_id
  AND ml.linked_movie_id = mc2.movie_id
  AND mi_idx2.movie_id = mc2.movie_id;

