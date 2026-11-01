/*+ NestLoop(k mk cn it1 mi mc t ct at)
 MergeJoin(k mk cn it1 mi mc t ct)
 HashJoin(mk cn it1 mi mc t ct)
 NestLoop(cn it1 mi mc t ct)
 NestLoop(cn it1 mi mc t)
 HashJoin(cn it1 mi mc)
 NestLoop(it1 mi mc)
 MergeJoin(it1 mi)
 IndexScan(k)
 SeqScan(mk)
 SeqScan(cn)
 SeqScan(it1)
 SeqScan(mi)
 IndexScan(mc)
 IndexScan(t)
 SeqScan(ct)
 IndexScan(at)
 Leading(((k (mk (((cn ((it1 mi) mc)) t) ct))) at)) */
SELECT MIN(at.title) AS aka_title,
       MIN(t.title) AS internet_movie_title
FROM aka_title AS at,
     company_name AS cn,
     company_type AS ct,
     info_type AS it1,
     keyword AS k,
     movie_companies AS mc,
     movie_info AS mi,
     movie_keyword AS mk,
     title AS t
WHERE cn.country_code = '[us]'
  AND it1.info = 'release dates'
  AND mi.note LIKE '%internet%'
  AND t.production_year > 1990
  AND t.id = at.movie_id
  AND t.id = mi.movie_id
  AND t.id = mk.movie_id
  AND t.id = mc.movie_id
  AND mk.movie_id = mi.movie_id
  AND mk.movie_id = mc.movie_id
  AND mk.movie_id = at.movie_id
  AND mi.movie_id = mc.movie_id
  AND mi.movie_id = at.movie_id
  AND mc.movie_id = at.movie_id
  AND k.id = mk.keyword_id
  AND it1.id = mi.info_type_id
  AND cn.id = mc.company_id
  AND ct.id = mc.company_type_id;

