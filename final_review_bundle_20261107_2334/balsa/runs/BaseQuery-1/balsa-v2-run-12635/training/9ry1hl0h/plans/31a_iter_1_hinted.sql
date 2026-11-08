/*+ HashJoin(n cn mc it2 t it1 mi mi_idx mk k ci)
 HashJoin(cn mc it2 t it1 mi mi_idx mk k ci)
 NestLoop(cn mc it2 t it1 mi mi_idx mk k)
 NestLoop(cn mc it2 t it1 mi mi_idx mk)
 HashJoin(cn mc it2 t it1 mi mi_idx)
 HashJoin(mc it2 t it1 mi mi_idx)
 HashJoin(it2 t it1 mi mi_idx)
 HashJoin(t it1 mi mi_idx)
 HashJoin(t it1 mi)
 HashJoin(it1 mi)
 SeqScan(n)
 SeqScan(cn)
 SeqScan(mc)
 SeqScan(it2)
 SeqScan(t)
 SeqScan(it1)
 SeqScan(mi)
 IndexScan(mi_idx)
 IndexScan(mk)
 IndexScan(k)
 SeqScan(ci)
 Leading((n ((((cn (mc (it2 ((t (it1 mi)) mi_idx)))) mk) k) ci))) */
SELECT MIN(mi.info) AS movie_budget,
       MIN(mi_idx.info) AS movie_votes,
       MIN(n.name) AS writer,
       MIN(t.title) AS violent_liongate_movie
FROM cast_info AS ci,
     company_name AS cn,
     info_type AS it1,
     info_type AS it2,
     keyword AS k,
     movie_companies AS mc,
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
  AND cn.name LIKE 'Lionsgate%'
  AND it1.info = 'genres'
  AND it2.info = 'votes'
  AND k.keyword IN ('murder',
                    'violence',
                    'blood',
                    'gore',
                    'death',
                    'female-nudity',
                    'hospital')
  AND mi.info IN ('Horror',
                  'Thriller')
  AND n.gender = 'm'
  AND t.id = mi.movie_id
  AND t.id = mi_idx.movie_id
  AND t.id = ci.movie_id
  AND t.id = mk.movie_id
  AND t.id = mc.movie_id
  AND ci.movie_id = mi.movie_id
  AND ci.movie_id = mi_idx.movie_id
  AND ci.movie_id = mk.movie_id
  AND ci.movie_id = mc.movie_id
  AND mi.movie_id = mi_idx.movie_id
  AND mi.movie_id = mk.movie_id
  AND mi.movie_id = mc.movie_id
  AND mi_idx.movie_id = mk.movie_id
  AND mi_idx.movie_id = mc.movie_id
  AND mk.movie_id = mc.movie_id
  AND n.id = ci.person_id
  AND it1.id = mi.info_type_id
  AND it2.id = mi_idx.info_type_id
  AND k.id = mk.keyword_id
  AND cn.id = mc.company_id;

