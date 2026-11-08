/*+ HashJoin(cn k mk t at it1 mi mc ct)
 HashJoin(k mk t at it1 mi mc ct)
 HashJoin(mc ct)
 HashJoin(k mk t at it1 mi)
 HashJoin(it1 mi)
 HashJoin(k mk t at)
 HashJoin(t at)
 HashJoin(k mk)
 SeqScan(cn)
 IndexScan(k)
 SeqScan(mk)
 SeqScan(t)
 SeqScan(at)
 SeqScan(it1)
 IndexScan(mi)
 IndexScan(mc)
 IndexScan(ct)
 Leading((cn ((((k mk) (t at)) (it1 mi)) (mc ct)))) */
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

