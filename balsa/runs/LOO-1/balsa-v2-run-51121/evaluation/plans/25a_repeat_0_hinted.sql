/*+ HashJoin(it1 mi k mk n ci t it2 mi_idx)
 HashJoin(mi k mk n ci t it2 mi_idx)
 MergeJoin(k mk n ci t it2 mi_idx)
 HashJoin(mk n ci t it2 mi_idx)
 MergeJoin(n ci t it2 mi_idx)
 HashJoin(ci t it2 mi_idx)
 MergeJoin(t it2 mi_idx)
 HashJoin(it2 mi_idx)
 IndexScan(it1)
 SeqScan(mi)
 IndexScan(k)
 SeqScan(mk)
 IndexScan(n)
 SeqScan(ci)
 IndexScan(t)
 IndexScan(it2)
 IndexScan(mi_idx)
 Leading((it1 (mi (k (mk (n (ci (t (it2 mi_idx))))))))) */
SELECT MIN(mi.info) AS movie_budget,
       MIN(mi_idx.info) AS movie_votes,
       MIN(n.name) AS male_writer,
       MIN(t.title) AS violent_movie_title
FROM cast_info AS ci,
     info_type AS it1,
     info_type AS it2,
     keyword AS k,
     movie_info AS mi,
     movie_info_idx AS mi_idx,
     movie_keyword AS mk,
     name AS n,
     title AS t
WHERE ci.note IN ('(writer)',
                  '(head writer)',
                  '(written by)',
                  '(story)',
                  '(story editor)')
  AND it1.info = 'genres'
  AND it2.info = 'votes'
  AND k.keyword IN ('murder',
                    'blood',
                    'gore',
                    'death',
                    'female-nudity')
  AND mi.info = 'Horror'
  AND n.gender = 'm'
  AND t.id = mi.movie_id
  AND t.id = mi_idx.movie_id
  AND t.id = ci.movie_id
  AND t.id = mk.movie_id
  AND ci.movie_id = mi.movie_id
  AND ci.movie_id = mi_idx.movie_id
  AND ci.movie_id = mk.movie_id
  AND mi.movie_id = mi_idx.movie_id
  AND mi.movie_id = mk.movie_id
  AND mi_idx.movie_id = mk.movie_id
  AND n.id = ci.person_id
  AND it1.id = mi.info_type_id
  AND it2.id = mi_idx.info_type_id
  AND k.id = mk.keyword_id;

