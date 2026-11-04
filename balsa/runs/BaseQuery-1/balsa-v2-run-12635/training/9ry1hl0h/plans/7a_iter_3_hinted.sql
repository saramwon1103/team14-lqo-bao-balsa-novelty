/*+ HashJoin(an n ci lt t ml pi it)
 NestLoop(n ci lt t ml pi it)
 NestLoop(n ci lt t ml pi)
 HashJoin(n ci lt t ml)
 HashJoin(ci lt t ml)
 NestLoop(lt t ml)
 NestLoop(t ml)
 IndexScan(an)
 SeqScan(n)
 SeqScan(ci)
 SeqScan(lt)
 SeqScan(t)
 IndexScan(ml)
 IndexScan(pi)
 SeqScan(it)
 Leading((an (((n (ci (lt (t ml)))) pi) it))) */
SELECT MIN(n.name) AS of_person,
       MIN(t.title) AS biography_movie
FROM aka_name AS an,
     cast_info AS ci,
     info_type AS it,
     link_type AS lt,
     movie_link AS ml,
     name AS n,
     person_info AS pi,
     title AS t
WHERE an.name LIKE '%a%'
  AND it.info ='mini biography'
  AND lt.link ='features'
  AND n.name_pcode_cf BETWEEN 'A' AND 'F'
  AND (n.gender='m'
       OR (n.gender = 'f'
           AND n.name LIKE 'B%'))
  AND pi.note ='Volker Boehm'
  AND t.production_year BETWEEN 1980 AND 1995
  AND n.id = an.person_id
  AND n.id = pi.person_id
  AND ci.person_id = n.id
  AND t.id = ci.movie_id
  AND ml.linked_movie_id = t.id
  AND lt.id = ml.link_type_id
  AND it.id = pi.info_type_id
  AND pi.person_id = an.person_id
  AND pi.person_id = ci.person_id
  AND an.person_id = ci.person_id
  AND ci.movie_id = ml.linked_movie_id;

